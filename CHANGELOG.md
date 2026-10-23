# Changelog

## 0.1.0 (unreleased)

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

After the independent review (2026-10-24):

README.md
---------
1. MCP section (line 53-54), replace "The server reads and writes only inside `--root` (default: the folder it
   starts in) and refuses other paths with a clear error." with:
   "The server reads and writes only inside `--root` (default: the folder it starts in; it refuses to start when
   that is a drive or file-system root or your home folder). Paths are checked before any file access: UNC and
   device paths (\\server\share, \\?\, \\.\, //host), other drives and anything `..` takes outside the root are
   refused with one message. Exports never replace an existing file unless the call passes `overwrite: true`,
   and never replace the open image. It prints the root it serves on stderr at start-up."
2. CLI section: add "Outputs that already exist are refused unless you pass `--force`; the input image is never
   overwritten."
3. Find / tiles: add "One find uses at most 64 tiles and a 48x48 grid; past that it says how many tiles your
   settings give, about how long they would take, and which tile size fits. A tiled find has no whole-image
   pass: objects larger than about a third of a tile are dropped, so run a plain find first for big objects."
4. Install / models (line 22-30): add "Maskbolt keeps the checked .pth next to the converted safetensors and
   records the converted file's SHA-256 in `<name>.safetensors.sha256`; every load re-checks it and rebuilds the
   file from the .pth if it does not match. Several Maskbolt processes can share the folder (unique temp names,
   atomic renames). Downloads give up after 30 s without a connection and 60 s without a response."
   Disk use per model is now the checkpoint plus the converted file (ViT-B about 750 MB, MobileSAM about 81 MB).
   A cache made by the first build (converted file, no sidecar, .pth deleted) is rebuilt once on first use,
   which downloads the checkpoint again.
5. Images: "Images are recognised by their content, not their extension, and EXIF orientation is applied.
   The limit is 100 megapixels."
6. Exports: "JSON names the image by file name only. SVG and JSON keep 0.01 px of precision at any mm per pixel
   (more decimals for small mm_per_px); area_mm2 has six significant digits."
7. MCP view: "view replies include the formula image_x = region.x + preview_x / scale; previews are at most
   1568 px on the long side."
8. Limits: replace "Long tiled finds can outlast an MCP client's tool timeout (the 400 px corn run takes 3.5
   minutes)." with "Long tiled finds can outlast an MCP client's tool timeout (the 400 px corn run, 25 tiles,
   takes 3.5 minutes; the cap of 64 tiles is about 4 to 10 minutes). There are no progress notifications yet."
9. Licence (line 90): "MIT. Model weights: Apache-2.0 (Meta's SAM; MobileSAM). src/sam_decoder.rs is adapted
   from candle (MIT OR Apache-2.0); candle's licence texts are in THIRD_PARTY.txt."
10. Toolchain: the minimum Rust is now 1.89 (Cargo.toml rust-version).

CHANGELOG.md (under 0.1.0, or a new "Unreleased" section)
---------------------------------------------------------
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
