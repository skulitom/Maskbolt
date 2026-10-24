// Dependency licence notices for one build target, from Cargo's target-filtered graph and the crate sources Cargo
// already fetched. Generated texts belong in build output only; never commit them or crate sources.
// Adapted from skulitom/Cutbolt's MIT-licensed mcpb/notices.mjs.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { root, writeUnchangedOrNew } from './build.mjs';

const compare = (a, b) => a < b ? -1 : a > b ? 1 : 0;
const noticeName = /^(?:licen[cs]e|copying|notice|copyright|unlicense)(?:[-._].*)?$/i;
const cell = value => String(value ?? '').replaceAll('|', '\\|').replace(/[\r\n]+/g, ' ');

// Crates published without a licence file, mapped to the crate from the same repository, under the same licence,
// whose files carry the texts. Used only when both are in the graph with equal repository and licence fields.
export const SHARED_TEXTS = {
  'candle-nn': 'candle-core',
  'candle-transformers': 'candle-core',
  'pulp-wasm-simd-flag': 'pulp'
};

export function normalPackages(metadata) {
  const packages = new Map(metadata.packages.map(pkg => [pkg.id, pkg]));
  const nodes = new Map(metadata.resolve?.nodes?.map(node => [node.id, node]));
  const rootId = metadata.resolve?.root;
  if (!rootId || packages.get(rootId)?.name !== 'maskbolt') throw new Error('Cargo metadata must resolve the root package maskbolt.');
  const visited = new Set(), pending = [rootId];
  while (pending.length) {
    const id = pending.pop();
    if (visited.has(id)) continue;
    visited.add(id);
    const node = nodes.get(id);
    if (!node || !packages.has(id)) throw new Error(`Incomplete Cargo dependency graph: ${id}`);
    for (const dep of node.deps) {
      if (dep.dep_kinds.some(kind => kind.kind === null)) pending.push(dep.pkg);
    }
  }
  return [...visited].filter(id => id !== rootId).map(id => packages.get(id))
    .sort((a, b) => compare(a.name, b.name) || compare(a.version, b.version) || compare(a.id, b.id));
}

const normalRepo = url => String(url ?? '').trim().replace(/\.git$/, '').replace(/\/+$/, '').toLowerCase();

// Pure renderer with an injected reader: readFile(path, 'utf8') and readdir(path, { withFileTypes: true }).
// Tests need no Cargo or real crate files. Cargo already filters platforms; do not interpret dep_kinds[].target.
export async function renderNotices(metadata, reader) {
  const packages = normalPackages(metadata), texts = new Map(), rows = [], missing = [];
  const files = new Map();
  for (const pkg of packages) {
    const directory = path.dirname(pkg.manifest_path), found = new Set();
    try {
      if (pkg.license_file) found.add(path.resolve(directory, pkg.license_file));
      for (const entry of await reader.readdir(directory, { withFileTypes: true })) {
        if (entry.isFile() && noticeName.test(entry.name)) found.add(path.resolve(directory, entry.name));
      }
    } catch (error) {
      throw new Error(`Cannot read notices for ${pkg.name} ${pkg.version}: ${error.message}`, { cause: error });
    }
    files.set(pkg.id, { directory, found: [...found].sort(compare) });
  }
  for (const pkg of packages) {
    let { directory, found } = files.get(pkg.id);
    let via = '';
    if (!found.length && Object.hasOwn(SHARED_TEXTS, pkg.name)) {
      const donor = packages.find(other => other.name === SHARED_TEXTS[pkg.name] && files.get(other.id).found.length);
      if (donor && normalRepo(donor.repository) === normalRepo(pkg.repository) && normalRepo(donor.repository) && donor.license === pkg.license) {
        ({ directory, found } = files.get(donor.id));
        via = `${donor.name} ${donor.version} `;
      }
    }
    const notices = [];
    for (const file of found) {
      let text;
      try { text = String(await reader.readFile(file, 'utf8')).replace(/\r\n?/g, '\n').trimEnd(); } catch (error) {
        throw new Error(`Cannot read notices for ${pkg.name} ${pkg.version}: ${error.message}`, { cause: error });
      }
      if (!text) continue;
      const id = `notice-${createHash('sha256').update(text).digest('hex').slice(0, 16)}`;
      if (texts.has(id) && texts.get(id) !== text) throw new Error(`Notice hash collision: ${id}`);
      texts.set(id, text);
      notices.push(`${cell(via + path.relative(directory, file).split(path.sep).join('/'))}: [${id}](#${id})`);
    }
    if (!notices.length) missing.push(`${pkg.name} ${pkg.version}`);
    rows.push(`| ${cell(pkg.name)} | ${cell(pkg.version)} | ${cell(pkg.license)} | ${notices.join('; ')} |`);
  }
  if (missing.length) {
    const error = new Error(`No license text for: ${missing.join(', ')} (${packages.length} normal dependency packages checked).`);
    error.packages = missing;
    error.packageCount = packages.length;
    throw error;
  }
  const sections = [...texts].sort(([a], [b]) => compare(a, b)).map(([id, text]) => {
    // Preserve arbitrary text, even if a notice itself contains Markdown fences.
    const fence = '`'.repeat(Math.max(3, ...[...text.matchAll(/`+/g)].map(match => match[0].length + 1)));
    return `## ${id}\n\n${fence}text\n${text}\n${fence}`;
  });
  const text = '# Third-party license notices\n\n' +
    'These are the distributed license texts for Maskbolt\'s normal Rust dependencies for this build target. ' +
    'Versions and checksums are pinned in Cargo.lock; sources remain in the external Cargo cache. ' +
    'Shared texts appear once with stable references; a crate published without its own licence file points to ' +
    'the file of the crate from the same repository and licence that carries it. The model weights, which ' +
    'Maskbolt downloads on first use, and the adapted candle code are described in THIRD_PARTY.txt.\n\n' +
    '| Package | Version | License expression | Notices |\n| --- | --- | --- | --- |\n' +
    rows.join('\n') + '\n\n' + sections.join('\n\n') + '\n';
  return { text, packageCount: packages.length, noticeCount: texts.size };
}

async function main(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (!['--target', '--out'].includes(key) || !argv[i + 1] || argv[i + 1].startsWith('--') || Object.hasOwn(options, key)) {
      throw new Error('Usage: node mcpb/notices.mjs --target <triple> [--out <file>]');
    }
    options[key] = argv[++i];
  }
  const target = options['--target'];
  if (!target || !/^[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)+$/.test(target)) throw new Error('--target must be a target triple.');
  const metadata = JSON.parse(execFileSync('cargo', ['metadata', '--format-version', '1', '--locked', '--offline', '--filter-platform', target], {
    cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, windowsHide: true, stdio: ['ignore', 'pipe', 'inherit']
  }));
  const result = await renderNotices(metadata, { readFile, readdir });
  const out = path.resolve(options['--out'] ?? path.join(root, 'target/notices', target, 'THIRD_PARTY_NOTICES.md'));
  const bytes = Buffer.from(result.text);
  await writeUnchangedOrNew(out, bytes);
  console.log(`${target}: ${result.packageCount} packages, ${result.noticeCount} distinct texts, ${bytes.length} bytes\n${out}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1; });
}
