import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, mkdtemp, rm, link } from 'node:fs/promises';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { archive, TARGETS } from './archive.mjs';
import { root, cargoVersion, assembleMcpb, buildMcpb, registryEntry, BUNDLE_NAME, BUNDLE_FILES, json, mergeTools } from './build.mjs';
import { readZip, writeZip, sha256 } from './zip.mjs';
import { checkManifest, inside } from './verify.mjs';
import { renderNotices, SHARED_TEXTS } from './notices.mjs';

let temp, base, binary, notices, template, metadata, build, fixtureTools;
// Packaging fixtures need no executable; the release workflow's archive and bundle steps run the real one.
const assembleFixture = options => assembleMcpb({ notices, ...options }, async () => fixtureTools);
const buildFixture = options => buildMcpb({ notices, ...options }, async () => fixtureTools);
const version = '7.8.9';
const writeJSON = (file, value) => writeFile(file, json(value));
const cargo = value => `[package]\nname = "fixture"\nversion = "${value}"\n\n[dependencies]\nunused = "0.1.0"\n`;
const NOTICES = '# Third-party license notices\r\n\r\nFixture notices\r\n';

before(async () => {
  await mkdir(path.join(root, 'target'), { recursive: true });
  temp = await mkdtemp(path.join(root, 'target/mcpb-test-'));
  base = path.join(temp, 'source');
  await mkdir(path.join(base, 'mcpb'), { recursive: true });
  template = JSON.parse(await readFile(path.join(root, 'mcpb/manifest.json')));
  fixtureTools = [...template.tools.slice(1).reverse(), { name: 'maskbolt_new', description: 'New tool sentence. More detail.' }];
  metadata = { ...JSON.parse(await readFile(path.join(root, 'server.json'))), version: '0.0.1' };
  await writeJSON(path.join(base, 'mcpb/manifest.json'), template);
  await writeJSON(path.join(base, 'server.json'), metadata);
  await writeFile(path.join(base, 'Cargo.toml'), cargo(version));
  await writeFile(path.join(base, 'LICENSE'), 'Fixture license\r\n');
  await writeFile(path.join(base, 'THIRD_PARTY.txt'), 'Fixture third-party notes\r\n');
  await writeFile(path.join(base, 'README.md'), 'Fixture readme\r\n');
  notices = path.join(temp, 'THIRD_PARTY_NOTICES.md');
  await writeFile(notices, NOTICES);
  binary = path.join(base, 'stand-in.exe');
  await writeFile(binary, Buffer.from([77, 90, 0, 13, 10, 255]));
  build = await buildFixture({ base, binary, out: path.join(temp, 'first') });
});
after(async () => {
  if (temp) {
    assert.ok(inside(path.join(root, 'target'), temp));
    await rm(temp, { recursive: true, force: true });
  }
});

test('server.json stays package-free with the registry identity and a short description', async () => {
  const server = JSON.parse(await readFile(path.join(root, 'server.json')));
  assert.equal(server.name, 'io.github.skulitom/maskbolt');
  assert.equal(server.$schema, 'https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json');
  assert.equal(server.repository.url, 'https://github.com/skulitom/Maskbolt');
  assert.equal(server.repository.source, 'github');
  assert.equal(server.repository.id, '1432055660');
  assert.equal(server.version, await cargoVersion());
  assert.equal(Object.hasOwn(server, 'packages'), false);
  assert.ok(server.description.length > 0 && server.description.length <= 100, `description is ${server.description.length} characters`);
});

test('manifest template has no version and valid binary-server fields', async () => {
  assert.equal(Object.hasOwn(template, 'version'), false);
  checkManifest({ ...template, version: await cargoVersion() }, await cargoVersion());
});

test('manifest tools are exactly the tools the server source defines', async () => {
  const source = await readFile(path.join(root, 'src/mcp.rs'), 'utf8');
  const defined = [...source.matchAll(/#\[tool\(description[\s\S]*?\)\]\s*async fn (maskbolt_\w+)/g)].map(match => match[1]);
  assert.equal(defined.length, 8);
  assert.deepEqual(template.tools.map(tool => tool.name).sort(), defined.sort());
});

test('release targets match the workflow matrix', async () => {
  const workflow = await readFile(path.join(root, '.github/workflows/release.yml'), 'utf8');
  const matrix = [...workflow.matchAll(/target: ([\w-]+), binary: (maskbolt(?:\.exe)?)/g)].map(match => [match[1], match[2]]);
  assert.deepEqual(matrix.map(([target]) => target).sort(), Object.keys(TARGETS).sort());
  for (const [target, name] of matrix) assert.equal(TARGETS[target].binary, name);
});

test('tool merge keeps curated descriptions, drops absent tools and follows live order without mutation', () => {
  const overrides = [
    { name: 'maskbolt_first', description: 'Curated first.' },
    { name: 'maskbolt_dropped', description: 'Retired.' },
    { name: 'maskbolt_last', description: 'Curated last.' }
  ];
  const live = [
    { name: 'maskbolt_last', description: 'Server last.', inputSchema: {} },
    { name: 'maskbolt_first', description: 'Server first.' }
  ];
  const before = structuredClone({ overrides, live });
  assert.deepEqual(mergeTools(overrides, live), { tools: [overrides[2], overrides[0]], added: [], dropped: ['maskbolt_dropped'] });
  assert.deepEqual({ overrides, live }, before);
});

test('new tool descriptions use the first sentence, at most 160 characters on a word boundary', () => {
  const merge = description => mergeTools([], [{ name: 'maskbolt_new', description }]).tools[0].description;
  assert.equal(merge('  Find every\nobject. More detail.  '), 'Find every object.');
  assert.equal(merge('Zoom at 1.5 scale. More.'), 'Zoom at 1.5 scale.');
  const exact = 'word '.repeat(31) + 'final';
  assert.equal(merge(exact + ' more words.'), exact);
  assert.throws(() => merge('x'.repeat(161)), /word boundary/);
  for (const description of [undefined, null, '', ' \n ']) assert.throws(() => merge(description), /server description/);
});

test('tool merge rejects empty catalogs, invalid names and duplicates', () => {
  for (const live of [[], undefined, {}]) assert.throws(() => mergeTools([], live), /no tools/);
  for (const name of ['', 'maskbolt_', 'other_tool', 'maskbolt_bad-name', 'maskbolt_é', ' maskbolt_bad', 7, null]) {
    assert.throws(() => mergeTools([], [{ name, description: 'Description.' }]), /Invalid tool name/);
  }
  const tool = { name: 'maskbolt_valid_1', description: 'Description.' };
  assert.throws(() => mergeTools([], [tool, tool]), /Duplicate tool name/);
});

test('two builds are identical, with exactly the required entries and checksum', async () => {
  const second = await buildFixture({ base, binary, out: path.join(temp, 'second') });
  assert.deepEqual(second.bytes, build.bytes);
  const entries = readZip(build.bytes);
  assert.deepEqual([...entries.keys()], [...BUNDLE_FILES].sort());
  assert.deepEqual(entries.get('server/maskbolt.exe').read(), await readFile(binary));
  assert.equal(entries.get('LICENSE').read().toString(), 'Fixture license\n');
  assert.equal(entries.get('THIRD_PARTY.txt').read().toString(), 'Fixture third-party notes\n');
  assert.equal(entries.get('THIRD_PARTY_NOTICES.md').read().toString(), NOTICES.replaceAll('\r\n', '\n'));
  const generated = JSON.parse(entries.get('manifest.json').read());
  assert.equal(generated.version, version);
  assert.deepEqual(generated.tools, mergeTools(template.tools, fixtureTools).tools);
  assert.deepEqual(build.added, ['maskbolt_new']);
  assert.deepEqual(build.dropped, [template.tools[0].name]);
  assert.equal(await readFile(`${build.bundle}.sha256`, 'utf8'), `${sha256(build.bytes)}  ${BUNDLE_NAME}\n`);
  await buildFixture({ base, binary, out: path.join(temp, 'first') });
});

test('builder refuses a template version, invalid versions, missing binaries and missing notices', async () => {
  try {
    await writeJSON(path.join(base, 'mcpb/manifest.json'), { ...template, version });
    await assert.rejects(assembleFixture({ base, binary }), /must not contain version/);
  } finally { await writeJSON(path.join(base, 'mcpb/manifest.json'), template); }
  try {
    for (const value of ['1.0', 'v1.0.0', '01.0.0', '1.0.0-beta']) {
      await writeFile(path.join(base, 'Cargo.toml'), cargo(value));
      await assert.rejects(assembleFixture({ base, binary }), /version must have the form X.Y.Z/);
    }
  } finally { await writeFile(path.join(base, 'Cargo.toml'), cargo(version)); }
  await assert.rejects(assembleFixture({ base }), /--binary is required/);
  await assert.rejects(assembleFixture({ base, binary: path.join(temp, 'missing.exe') }), /Missing binary/);
  await assert.rejects(assembleMcpb({ base, binary }, async () => fixtureTools), /--notices is required/);
  await assert.rejects(assembleFixture({ base, binary, notices: path.join(base, 'README.md') }), /not a generated notices file/);
});

test('existing different artifacts are preserved', async () => {
  const out = path.join(temp, 'occupied'); await mkdir(out);
  const file = path.join(out, BUNDLE_NAME); await writeFile(file, 'keep');
  await assert.rejects(buildFixture({ base, binary, out }), /already exists with different contents/);
  assert.equal(await readFile(file, 'utf8'), 'keep');
});

test('native archives are deterministic and carry the binary, readme, licence and notices', async () => {
  for (const target of Object.keys(TARGETS)) {
    const options = { base, binary, notices, target, smoke: false, out: path.join(temp, 'archives') };
    const first = await archive(options), second = await archive(options);
    assert.deepEqual(first.bytes, second.bytes);
    assert.equal(path.basename(first.file), `maskbolt-v${version}-${target}.${TARGETS[target].extension}`);
    const isWindows = target.includes('windows'), entries = new Map();
    if (isWindows) {
      for (const [name, entry] of readZip(first.bytes)) entries.set(name, entry.read());
    } else {
      const tar = gunzipSync(first.bytes);
      const string = (start, size) => tar.subarray(start, start + size).toString().replace(/\0.*$/s, '');
      for (let at = 0; tar[at];) {
        const name = string(at, 100), size = parseInt(string(at + 124, 12), 8);
        assert.equal(string(at + 257, 5), 'ustar');
        assert.equal(parseInt(string(at + 100, 8), 8), name === 'maskbolt' ? 0o755 : 0o644);
        const header = Buffer.from(tar.subarray(at, at + 512)), sum = parseInt(string(at + 148, 8), 8);
        header.fill(32, 148, 156);
        assert.equal(header.reduce((a, b) => a + b, 0), sum);
        entries.set(name, tar.subarray(at + 512, at + 512 + size));
        at += 512 + Math.ceil(size / 512) * 512;
      }
    }
    const executable = TARGETS[target].binary;
    assert.deepEqual([...entries.keys()].sort(), ['LICENSE', 'README.md', 'THIRD_PARTY.txt', 'THIRD_PARTY_NOTICES.md', executable].sort());
    assert.deepEqual(entries.get(executable), await readFile(binary));
    assert.equal(entries.get('README.md').toString(), 'Fixture readme\n');
  }
  await assert.rejects(archive({ base, binary, notices, target: 'riscv64gc-unknown-linux-gnu', smoke: false }), /Unsupported release target/);
});

test('registry entry is installable: the MCPB package with its SHA-256, and inputs are untouched', async () => {
  const out = path.join(temp, 'server.registry.json'), before = await readFile(path.join(base, 'server.json'));
  const entry = await registryEntry({ base, tag: `v${version}`, bundle: build.bundle, out });
  assert.equal(entry.version, version);
  assert.deepEqual(entry.packages, [{ registryType: 'mcpb', identifier: `https://github.com/skulitom/Maskbolt/releases/download/v${version}/${BUNDLE_NAME}`, version, fileSha256: sha256(build.bytes), transport: { type: 'stdio' } }]);
  assert.deepEqual(JSON.parse(await readFile(out)), entry);
  assert.deepEqual(await readFile(path.join(base, 'server.json')), before);
  await registryEntry({ base, tag: `v${version}`, bundle: build.bundle, out });
});

test('registry entry rejects bad tags, version mismatches, other names and packages in metadata', async () => {
  const options = { base, tag: `v${version}`, bundle: build.bundle, out: path.join(temp, 'refused.json') };
  for (const tag of ['7.8.9', 'v1.2', 'v01.2.3', 'v1.2.3-beta', '../v1.2.3', 'v1.2.3\n']) await assert.rejects(registryEntry({ ...options, tag }), /Tag must have the form/);
  await assert.rejects(registryEntry({ ...options, tag: 'v7.8.10' }), /Tag and Cargo.toml versions must match/);
  try {
    await writeJSON(path.join(base, 'server.json'), { ...metadata, packages: [] });
    await assert.rejects(registryEntry(options), /metadata-only/);
    await writeJSON(path.join(base, 'server.json'), { ...metadata, name: 'io.github.skulitom/other' });
    await assert.rejects(registryEntry(options), /must name io.github.skulitom\/maskbolt/);
  } finally { await writeJSON(path.join(base, 'server.json'), metadata); }
  const files = new Map([...readZip(build.bytes)].map(([name, entry]) => [name, entry.read()]));
  files.set('manifest.json', json({ ...build.manifest, version: '7.8.8' }));
  const oldBundle = path.join(temp, 'old.mcpb'); await writeFile(oldBundle, writeZip(files));
  await assert.rejects(registryEntry({ ...options, bundle: oldBundle }), /Bundle manifest version must match/);
});

test('registry output cannot overwrite metadata or the bundle, including hard-link aliases', async () => {
  const alias = path.join(temp, 'alias.json'); await link(path.join(base, 'server.json'), alias);
  const outputs = [path.join(base, 'server.json'), build.bundle, alias, path.join(base, 'mcpb/../server.json')];
  if (process.platform === 'win32') outputs.push(path.join(base, 'SERVER.JSON'));
  for (const out of outputs) {
    const before = await readFile(out);
    await assert.rejects(registryEntry({ base, tag: `v${version}`, bundle: build.bundle, out }), /must not overwrite/);
    assert.deepEqual(await readFile(out), before);
  }
});

test('ZIP reader rejects malformed archives and detects CRC corruption', () => {
  assert.throws(() => readZip(Buffer.from('no')), /Not a ZIP/);
  const bytes = writeZip(new Map([['small', Buffer.from('abc')]]));
  bytes[30 + Buffer.byteLength('small')] ^= 1;
  assert.throws(() => readZip(bytes).get('small').read(), /damaged/);
});

test('verifier rejects other versions, launch commands, platforms and a default root', () => {
  assert.throws(() => checkManifest({ ...template, version: '1.0.0' }, '1.0.1'), /version must equal/);
  assert.throws(() => checkManifest({ ...build.manifest, compatibility: { platforms: ['linux'] } }, version));
  assert.throws(() => checkManifest({ ...build.manifest, server: { ...template.server, mcp_config: { command: 'maskbolt', args: ['mcp'] } } }, version));
  assert.throws(() => checkManifest({ ...build.manifest, user_config: { root: { ...template.user_config.root, default: '${HOME}' } } }, version), /must not have a default/);
});

test('notices: shared texts only from the same repository and licence; missing texts fail', async () => {
  const pkg = (name, license, repository) => ({ id: `${name} 1.0.0`, name, version: '1.0.0', license, repository, manifest_path: `/crates/${name}/Cargo.toml` });
  const metadata = (packages) => ({
    packages: [{ id: 'maskbolt 0.1.0', name: 'maskbolt', version: '0.1.0', manifest_path: '/src/Cargo.toml' }, ...packages],
    resolve: { root: 'maskbolt 0.1.0', nodes: [
      { id: 'maskbolt 0.1.0', deps: packages.map(p => ({ pkg: p.id, dep_kinds: [{ kind: null }] })) },
      ...packages.map(p => ({ id: p.id, deps: [] }))
    ] }
  });
  const files = { '/crates/candle-core': ['LICENSE'] };
  const reader = {
    readdir: async dir => (files[dir.replaceAll('\\', '/').replace(/^[A-Z]:/, '')] ?? []).map(name => ({ name, isFile: () => true })),
    readFile: async () => 'Apache License\nVersion 2.0'
  };
  assert.equal(SHARED_TEXTS['candle-nn'], 'candle-core');
  const candle = 'https://github.com/huggingface/candle';
  const ok = await renderNotices(metadata([pkg('candle-core', 'MIT OR Apache-2.0', candle), pkg('candle-nn', 'MIT OR Apache-2.0', `${candle}.git`)]), reader);
  assert.match(ok.text, /\| candle-nn \| 1\.0\.0 \| MIT OR Apache-2\.0 \| candle-core 1\.0\.0 LICENSE: \[notice-/);
  assert.equal(ok.noticeCount, 1);
  await assert.rejects(renderNotices(metadata([pkg('candle-core', 'MIT OR Apache-2.0', candle), pkg('candle-nn', 'MIT', candle)]), reader), /No license text for: candle-nn/);
  await assert.rejects(renderNotices(metadata([pkg('candle-core', 'MIT OR Apache-2.0', candle), pkg('candle-nn', 'MIT OR Apache-2.0', 'https://example.com/fork')]), reader), /No license text for: candle-nn/);
  await assert.rejects(renderNotices(metadata([pkg('lonely', 'MIT', 'https://example.com/lonely')]), reader), /No license text for: lonely/);
});
