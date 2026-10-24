// Native release archives with fixed metadata, after a smoke test of the binary. Node 20+, no external archiver.
import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { cargoVersion, readNotices, root, textFile, writeUnchangedOrNew } from './build.mjs';
import { writeZip } from './zip.mjs';
import { inside, mcpFolders, modelFreeChecks, startMcp } from './mcp-client.mjs';

// `hosts` lists the Node platform/arch pairs that can run the binary for the smoke test: the Intel macOS build is
// made on an Apple silicon runner and runs there through Rosetta 2.
export const TARGETS = {
  'x86_64-pc-windows-msvc': { hosts: ['win32-x64'], binary: 'maskbolt.exe', extension: 'zip' },
  'x86_64-unknown-linux-gnu': { hosts: ['linux-x64'], binary: 'maskbolt', extension: 'tar.gz' },
  'aarch64-apple-darwin': { hosts: ['darwin-arm64'], binary: 'maskbolt', extension: 'tar.gz' },
  'x86_64-apple-darwin': { hosts: ['darwin-x64', 'darwin-arm64'], binary: 'maskbolt', extension: 'tar.gz' }
};
export const ARCHIVE_FILES = ['LICENSE', 'README.md', 'THIRD_PARTY.txt'];

// Short root-level names fit the ustar name field; preserve the executable bit.
export function writeTarGz(files, executable) {
  const blocks = [], octal = (value, length) => value.toString(8).padStart(length - 1, '0') + '\0';
  for (const [name, data] of [...files].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) {
    if (!/^[A-Za-z0-9_.-]{1,100}$/.test(name)) throw new Error(`Unsupported tar entry name: ${name}`);
    if (data.length >= 8 ** 11) throw new Error(`Tar entry too large: ${name}`);
    const header = Buffer.alloc(512);
    header.write(name); header.write(octal(name === executable ? 0o755 : 0o644, 8), 100);
    header.write(octal(0, 8), 108); header.write(octal(0, 8), 116);
    header.write(octal(data.length, 12), 124); header.write(octal(315532800, 12), 136);
    header.fill(32, 148, 156); header.write('0', 156); header.write('ustar\0' + '00', 257);
    header.write(octal(header.reduce((sum, byte) => sum + byte, 0), 7) + ' ', 148);
    blocks.push(header, data, Buffer.alloc((512 - data.length % 512) % 512));
  }
  blocks.push(Buffer.alloc(1024));
  const bytes = gzipSync(Buffer.concat(blocks), { level: 9 });
  bytes[9] = 3; // Fixed gzip OS marker (Unix).
  return bytes;
}

// `maskbolt version`, then the MCP server: initialize, the full tool catalog and the model-free checks.
export async function smokeTest({ target, binary, base = root }) {
  const spec = TARGETS[target];
  if (!spec.hosts.includes(`${process.platform}-${process.arch}`)) throw new Error(`Smoke-test ${target} on a runner that can run it (${spec.hosts.join(' or ')}).`);
  const version = await cargoVersion(base);
  const result = spawnSync(path.resolve(binary), ['version'], { cwd: base, encoding: 'utf8', timeout: 15000, windowsHide: true });
  if (result.error) throw result.error;
  if (result.status !== 0 || result.stdout !== `maskbolt ${version}\n`) throw new Error(`maskbolt version printed ${JSON.stringify(result.stdout)} (exit ${result.status}): ${result.stderr}`);
  const tempRoot = path.resolve(base, 'target');
  await mkdir(tempRoot, { recursive: true });
  const temp = await mkdtemp(path.join(tempRoot, 'archive-smoke-'));
  let client;
  try {
    client = startMcp({ binary, ...await mcpFolders(temp) });
    await client.initialize(version);
    const tools = await client.listTools();
    await modelFreeChecks(client);
    return { version, tools: tools.length };
  } finally {
    try { await client?.stop(); } finally {
      if (!inside(tempRoot, temp)) throw new Error('Refusing to remove a folder outside target/');
      await rm(temp, { recursive: true, force: true });
    }
  }
}

export async function archive({ target, binary, notices, base = root, out = path.join(root, 'target/dist/release'), smoke = true }) {
  const spec = TARGETS[target];
  if (!spec) throw new Error(`Unsupported release target: ${target}`);
  if (!binary) throw new Error('--binary is required.');
  const checked = smoke ? await smokeTest({ target, binary, base }) : null;
  const files = new Map([[spec.binary, await readFile(binary)]]);
  for (const name of ARCHIVE_FILES) files.set(name, await textFile(path.join(base, name)));
  files.set('THIRD_PARTY_NOTICES.md', await readNotices(notices));
  const bytes = spec.extension === 'zip' ? writeZip(files) : writeTarGz(files, spec.binary);
  const file = path.resolve(out, `maskbolt-v${await cargoVersion(base)}-${target}.${spec.extension}`);
  await writeUnchangedOrNew(file, bytes);
  return { file, bytes, checked };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  Promise.resolve().then(async () => {
    const options = {}, args = process.argv.slice(2);
    for (let i = 0; i < args.length; i += 2) {
      if (!['--target', '--binary', '--notices', '--out'].includes(args[i]) || !args[i + 1] || args[i + 1].startsWith('--') || Object.hasOwn(options, args[i].slice(2))) throw new Error(`Unknown or incomplete option: ${args[i]}`);
      options[args[i].slice(2)] = args[i + 1];
    }
    const result = await archive(options);
    console.log(`${result.file}: ${result.bytes.length} bytes; version and MCP smoke test passed (${result.checked.tools} tools).`);
  }).catch(error => { console.error(error.message); process.exitCode = 1; });
}
