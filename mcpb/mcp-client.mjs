// Small stdio MCP client shared by tool discovery, archive smoke tests and bundle verification. Node built-ins only.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import path from 'node:path';

export const PROTOCOL = '2025-11-25';

export const inside = (base, file) => {
  const relative = path.relative(base, file);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
};

// Fresh folders for one server run: the root it serves, its working folder, a stand-in home (which also holds
// the model cache folder, so a real cache is never touched) and a temp folder.
export async function mcpFolders(temp) {
  const folders = Object.fromEntries(['root', 'cwd', 'home', 'scratch'].map(name => [name, path.join(temp, name === 'scratch' ? 'tmp' : name)]));
  for (const directory of Object.values(folders)) await mkdir(directory);
  return folders;
}

// Starts the server the way the manifest does (or `binary mcp --root <root>` when no manifest is given).
export function startMcp({ manifest, extension, binary, root, cwd, home, scratch, timeoutMs = 15000 }) {
  const config = manifest?.server?.mcp_config ?? { command: binary, args: ['mcp', '--root', '${user_config.root}'] };
  const substitute = value => value.replaceAll('${__dirname}', extension ?? '').replaceAll('${user_config.root}', root);
  // During a build, the supplied executable stands in for the packaged entry point.
  if (binary && manifest) assert.equal(config.command, '${__dirname}/' + manifest.server.entry_point);
  const pending = new Map();
  let id = 0, buffer = '', stderr = '', failure;
  const fail = error => { failure ??= error; for (const request of pending.values()) request.reject(error); pending.clear(); };
  const env = { ...process.env, HOME: home, USERPROFILE: home, APPDATA: home, LOCALAPPDATA: home, XDG_CACHE_HOME: path.join(home, '.cache'), TMP: scratch, TEMP: scratch, TMPDIR: scratch };
  // Never let a smoke test use (or fill) a real model cache.
  env.MASKBOLT_MODEL_DIR = path.join(home, 'maskbolt-models');
  const child = spawn(binary ? path.resolve(binary) : substitute(config.command), config.args.map(substitute), {
    cwd, env, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true
  });
  const closed = new Promise(resolve => child.once('close', (code, signal) => { fail(new Error(`MCP server exited (${code ?? signal}): ${stderr}`)); resolve(); }));
  child.on('error', fail); child.stdin.on('error', fail);
  child.stderr.setEncoding('utf8'); child.stderr.on('data', data => { stderr = (stderr + data).slice(-8000); });
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', data => {
    try {
      buffer += data;
      assert.ok(buffer.length < 8 * 1024 * 1024, 'MCP response exceeds the discovery/verification budget');
      let end;
      while ((end = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
        if (!line.trim()) continue;
        const message = JSON.parse(line), request = pending.get(message.id);
        assert.equal(message.jsonrpc, '2.0');
        if (!request) continue;
        pending.delete(message.id);
        if (message.error) request.reject(new Error(`${request.method}: ${JSON.stringify(message.error)}`));
        else request.resolve(message.result);
      }
    } catch (error) { fail(error); }
  });
  const timer = setTimeout(() => { fail(new Error(`MCP server timed out after ${timeoutMs / 1000} seconds. ${stderr}`)); child.kill(); }, timeoutMs);
  const send = message => {
    if (failure) throw failure;
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', ...message }) + '\n');
  };
  const request = (method, params) => new Promise((resolve, reject) => {
    if (failure) { reject(failure); return; }
    const current = ++id; pending.set(current, { resolve, reject, method });
    try { send({ id: current, method, params }); } catch (error) { fail(error); }
  });
  return {
    request,
    stderr: () => stderr,
    async initialize(version) {
      const initialized = await request('initialize', { protocolVersion: PROTOCOL, capabilities: {}, clientInfo: { name: 'maskbolt-mcpb', version: '1.0.0' } });
      assert.equal(typeof initialized.protocolVersion, 'string', 'initialize must return a protocol version');
      assert.equal(initialized.serverInfo?.name, 'maskbolt');
      if (version) assert.equal(initialized.serverInfo.version, version, 'binary version must equal the Cargo.toml version');
      assert.ok(initialized.capabilities?.tools, 'the server must offer tools');
      send({ method: 'notifications/initialized' });
      return initialized;
    },
    async listTools() {
      const tools = [], cursors = new Set();
      let cursor;
      do {
        const listed = await request('tools/list', cursor ? { cursor } : {});
        assert.ok(Array.isArray(listed.tools), 'tools/list must return tools');
        tools.push(...listed.tools);
        cursor = listed.nextCursor;
        if (cursor !== undefined && cursor !== null) {
          assert.ok(typeof cursor === 'string' && cursor && !cursors.has(cursor) && cursors.size < 100, 'Invalid or repeating catalog cursor');
          cursors.add(cursor);
        } else cursor = undefined;
      } while (cursor !== undefined);
      assert.ok(tools.length > 0, 'MCP server lists no tools');
      assert.equal(new Set(tools.map(tool => tool.name)).size, tools.length, 'duplicate live tool names');
      return tools;
    },
    async call(name, args) {
      const result = await request('tools/call', { name, arguments: args });
      const text = (result.content ?? []).filter(part => part.type === 'text').map(part => part.text).join('\n');
      return { ...result, text };
    },
    async stop() {
      clearTimeout(timer);
      child.kill();
      await closed;
    }
  };
}

// Calls that need no model, so they run anywhere without the 375 MB download: the root guard, a missing file and
// a tool used before any image is open. Each must fail cleanly with Maskbolt's own message.
export async function modelFreeChecks(client) {
  const outside = await client.call('maskbolt_open', { path: '../outside.png' });
  assert.equal(outside.isError, true, `a path outside the root must be refused: ${JSON.stringify(outside)}`);
  assert.match(outside.text, /outside the allowed root/);
  const missing = await client.call('maskbolt_open', { path: 'missing.png' });
  assert.equal(missing.isError, true, `a missing image must be an error: ${JSON.stringify(missing)}`);
  assert.match(missing.text, /was not found inside the allowed root/);
  const view = await client.call('maskbolt_view', {});
  assert.equal(view.isError, true, `view before open must be an error: ${JSON.stringify(view)}`);
  assert.match(view.text, /no image is open/);
}
