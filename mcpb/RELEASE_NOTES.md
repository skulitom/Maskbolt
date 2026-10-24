Maskbolt runs Meta's Segment Anything (SAM) on your CPU, for AI agents and scripts: one Rust
binary with an MCP server and a CLI. No Python, no GPU, no account.

### What's in 0.1.0

- **MCP server** (`maskbolt mcp --root <folder>`): eight tools to open an image, click an object or
  find every object, look at numbered previews, remove wrong masks, add missed ones, fill holes and
  export. Every reply that changes masks says how many there are, why candidates were dropped and
  which patches still look uncovered. Files are read and written only inside the root folder.
- **CLI**: `maskbolt cutout` (one object as a transparent PNG) and `maskbolt find` (every object, over
  the whole image or overlapping tiles for small objects), plus `fetch` and `version`.
- **Exports**: cutout PNG, numbered overlay PNG, SVG outlines with holes (even-odd, optionally in
  millimetres), JSON (outlines, area, box, score) and a labels PNG.
- **Models**: SAM ViT-B (default, 375 MB) or MobileSAM (`--model tiny`, 41 MB), both Apache-2.0,
  downloaded once from their official URLs on first use, SHA-256 checked and converted to
  safetensors in a user cache folder.

### Limits

CPU only: encoding takes about 6.5 s per image or tile with ViT-B on a 16-core desktop, longer on
smaller machines, and memory peaks at 2 to 3.3 GB for one image and about 7 GB for tiled finds.
Long tiled finds can outlast an MCP client's tool timeout. SAM misses thin parts and leaves holes.
See the [README](https://github.com/skulitom/Maskbolt#readme) for the measurements.

### Downloads

The Windows x64 ZIP and the Linux x64, macOS Apple silicon and macOS Intel tar.gz archives contain
`maskbolt` (`maskbolt.exe` on Windows), the README, the MIT licence, `THIRD_PARTY.txt` (model weights
and adapted code) and `THIRD_PARTY_NOTICES.md` (the licence texts of the Rust dependencies in that
build). Put `maskbolt` on your PATH; `maskbolt version` prints `maskbolt 0.1.0`. The Linux build
needs glibc 2.35 or newer.

`maskbolt-windows-x64.mcpb` is the Windows MCP server for Claude Desktop: Settings > Extensions >
Advanced settings > Install Extension, then choose your image folder.

For Claude Code, add the plugin, which registers the server and a skill (the `maskbolt` executable
must be on PATH):

```sh
claude plugin marketplace add skulitom/Maskbolt
claude plugin install maskbolt@maskbolt
```

Check downloads against `SHA256SUMS`: run `sha256sum --check --ignore-missing SHA256SUMS` in the
download folder, or on PowerShell compare `(Get-FileHash <file> -Algorithm SHA256).Hash` with its line.
Verify build provenance (substitute another asset name to check it):

```sh
gh attestation verify maskbolt-windows-x64.mcpb --repo skulitom/Maskbolt --signer-workflow skulitom/Maskbolt/.github/workflows/release.yml
```

MCP Registry publication is a separate step after testing; see
[Publishing](https://github.com/skulitom/Maskbolt/blob/main/mcpb/PUBLISHING.md).
