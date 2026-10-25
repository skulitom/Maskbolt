# Changelog

## Unreleased

- A Dockerfile builds a Linux image of the MCP server (non-root user, `/data` as the root, the model cached in
  `/cache`), for MCP catalogs that build from a Dockerfile; CI builds it and checks that the server lists its eight
  tools.
- `llms.txt` indexes the documentation for agents; the README opens with who Maskbolt is for, when to use it and when
  not, and how it compares with other MCP servers.

## 0.1.0 (2026-10-24)

First version: a Rust port of the 2026-10-23 Node prototype.

- Library: `Session` (open, click with include/exclude points, box and level; find over the image, a region or
  overlapping tiles with crop-edge and crop-background filters; dedupe; remove; fill holes; gaps and fill_gaps;
  remove_hole_views; numbered previews), a `Backend` trait with the candle SAM backend and a fake backend for
  tests. Masks are stored cropped to their boxes.
- Models: SAM ViT-B (default) and MobileSAM (`--model tiny`) from their official URLs, downloaded on first use,
  SHA-256 checked, converted once from .pth to safetensors in the user cache folder.
- Speed: the mask decoder's transposed convolutions run as matrix products, prompts decode on up to 16 threads
  (about 20 ms per prompt on 16 cores), tiles encode 4 at a time.
- MCP server (`maskbolt mcp --root DIR`): eight tools; replies carry counts, drop reasons, a "still uncovered"
  hint and PNG previews; paths are confined to the root.
- CLI: `cutout`, `find`, `mcp`, `fetch`, `version`, `help`.
- Exports: overlay PNG, cutout PNG, SVG (holes, even-odd, mm), JSON, labels PNG.
- Release: binaries for Windows x64 (static C runtime), Linux x64, macOS Apple silicon and macOS Intel with
  SHA256SUMS and build attestations, a Windows MCPB bundle for Claude Desktop, an MCP Registry entry
  (io.github.skulitom/maskbolt) and a Claude Code plugin with a maskbolt skill; CI runs tests and clippy on
  Windows, Linux and macOS.

Hardening after the independent review (2026-10-24):

- Fixed: exports (SVG, JSON, labels) hung for ever once a session had about 1,550 or more masks; label colours
  are now assigned in constant time per mask for any count.
- Security: the MCP root guard decides lexically before touching the file system (no SMB lookups for UNC paths,
  no existence oracle outside the root); drive roots and the home folder are refused as server roots.
- Exports no longer overwrite existing files by default (`overwrite: true` / `--force`), and never the open
  image; folders are created only when a file is written; existing links are not written through.
- find is capped at 64 tiles and a 48x48 grid, with an estimate in the error; each tile's embedding is freed
  once decoded.
- Model cache: unique temp names and atomic renames (safe across processes), the converted file's SHA-256 is
  recorded and checked on every load, a file that fails to load is rebuilt from the checked checkpoint,
  download timeouts and a size guard.
- candle's MIT and Apache-2.0 texts ship in THIRD_PARTY.txt for the adapted mask decoder.
- Smaller fixes: images open by content (a PNG named .jpg works) with EXIF orientation and a 100 MP cap; region
  arithmetic no longer overflows; click replies name the three levels exactly as `level` picks them; clearer
  tool descriptions; the MCP server survives a panic in a tool (mutex poisoning); a failed model download is
  retried on the next open.
- Fixed: an absolute path that named the MCP root as it was given, rather than as resolved, was refused as
  outside the root (macOS's /var -> /private/var temp folders, Windows 8.3 short names such as RUNNER~1, a root
  behind a link, junction or subst drive). Such paths are now rewritten onto the resolved root and checked as
  before; leaving the root with `..` is still refused.
- Tests: the 816-byte test checkpoint `tests/data/tiny.pth` is now committed (it was caught by the `*.pth`
  ignore rule, so `cargo test` failed in a fresh clone).
