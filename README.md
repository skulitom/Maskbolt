# Maskbolt

Segment Anything (SAM) on your CPU, for AI agents and scripts. Maskbolt is one Rust binary: an MCP server that
lets an agent open an image, find every object or click one, look at numbered previews, fix mistakes, and export
cutouts, outlines and masks; and a CLI for the same jobs in one command. No Python, no GPU.

- One click cuts an object out as a transparent PNG; the three nested SAM masks (part, object, group) are one
  argument away.
- "Find everything" runs grid prompts over the whole image or over overlapping tiles (each re-encoded at full
  model resolution, so small objects show up), drops masks cut by a tile edge, background patches, duplicates
  and masks inside others, and says why.
- Every reply that changes masks says how many there are, why candidates were dropped, and which patches still
  look uncovered ("find more like these"), so an agent knows what is left.
- Exports: overlay PNG (numbered masks, zoomable), cutout PNG, SVG (outer loops and holes, even-odd, optional
  millimetres), JSON (outlines with holes, area, box, score), labels PNG (one colour per mask).
- Masks are stored cropped to their boxes, so a dense image with thousands of candidates stays small.

## Install

Download the archive for your computer from the [latest release](https://github.com/skulitom/Maskbolt/releases/latest):
`maskbolt-v0.1.0-x86_64-pc-windows-msvc.zip`, `maskbolt-v0.1.0-x86_64-unknown-linux-gnu.tar.gz` (glibc 2.35 or
newer), `maskbolt-v0.1.0-aarch64-apple-darwin.tar.gz` or `maskbolt-v0.1.0-x86_64-apple-darwin.tar.gz`. Check it
against the release's `SHA256SUMS`, extract it and put `maskbolt` (`maskbolt.exe` on Windows) on your PATH;
`maskbolt version` prints `maskbolt 0.1.0`. Each archive also holds the licence, THIRD_PARTY.txt and
THIRD_PARTY_NOTICES.md (the licence texts of the Rust dependencies in that build). Or build it with Rust 1.89 or
newer:

    cargo install --git https://github.com/skulitom/Maskbolt --tag v0.1.0 --locked

Maskbolt is not on crates.io.

The first run downloads the model into a user cache folder and checks its SHA-256:

| Model | Flag | Download | Licence |
|---|---|---|---|
| SAM ViT-B (Meta) | default, `--model vit-b` | 375 MB | Apache-2.0 |
| MobileSAM (TinyViT) | `--model tiny` | 41 MB | Apache-2.0 |

Cache folder: `%LOCALAPPDATA%\maskbolt\models` on Windows, `$XDG_CACHE_HOME/maskbolt/models` or
`~/.cache/maskbolt/models` elsewhere; set `MASKBOLT_MODEL_DIR` to use another. `maskbolt fetch` downloads now.
Maskbolt keeps the checked `.pth` next to the converted safetensors and records the converted file's SHA-256 in
`<name>.safetensors.sha256`; every load re-checks it and rebuilds the file from the `.pth` if it does not match.
Several Maskbolt processes can share the folder (unique temp names, atomic renames). Disk use per model is the
checkpoint plus the converted file (ViT-B about 750 MB, MobileSAM about 81 MB). Minimum Rust: 1.89.

## CLI

    maskbolt cutout examples/chelsea.png --point 200,150 --out cat.png
    maskbolt cutout examples/coffee.png --point 300,225 --level largest --fill-holes --out cup-set.png
    maskbolt cutout photo.jpg --point 410,220 --exclude 380,300 --box 300,120,520,340 --out thing.png
    maskbolt find examples/corn.png --out kernels.json --overlay kernels.png
    maskbolt find examples/corn.png --tiles 400 --fill-gaps --out kernels.json --overlay kernels.png
    maskbolt find examples/bed.png --grid 12 --min-area 0.01 --max-area 0.3 --drop-hole-views \
        --out offcuts.json --svg offcuts.svg --mm-per-px 0.5
    maskbolt mcp --root C:\work\images
    maskbolt version

Outputs that already exist are refused unless you pass `--force`; the input image is never overwritten.
One find uses at most 64 tiles and a 48x48 grid; past that it says how many tiles your settings give, about how
long they would take, and which tile size fits. A tiled find has no whole-image pass: objects larger than about
a third of a tile are dropped, so run a plain find first for big objects. Images are recognised by their content,
not their extension, EXIF orientation is applied, and the limit is 100 megapixels.

`--level`: best (default, highest predicted IoU), smallest (a part), middle (the object), largest (the object
with what it holds). `find` prints a JSON report (counts, drop reasons, timings, what is still uncovered).

## MCP

    {"mcpServers": {"maskbolt": {"command": "maskbolt", "args": ["mcp", "--root", "C:\\work\\images"]}}}

- **Claude Code:** the plugin registers this server, rooted at the folder Claude Code runs in, and a `maskbolt`
  skill with the loop below. It needs `maskbolt` on PATH (see Install):

      claude plugin marketplace add skulitom/Maskbolt
      claude plugin install maskbolt@maskbolt

  or, inside Claude Code 2.1.275 or later, in one step: `/plugin install maskbolt --marketplace skulitom/Maskbolt`.

- **Claude Desktop (Windows x64):** download `maskbolt-windows-x64.mcpb` from the release, double-click it (or
  Settings > Extensions > Advanced settings > Install Extension) and choose your image folder.
- **MCP Registry:** listed as `io.github.skulitom/maskbolt`, with the Windows bundle and its SHA-256.

Long tiled finds take minutes: the plugin gives Maskbolt's tools a 10-minute timeout; other clients may stop
waiting sooner (see Limits).

Tools: `maskbolt_open`, `maskbolt_find`, `maskbolt_click`, `maskbolt_view`, `maskbolt_remove`, `maskbolt_gaps`
(`add: true` clicks the candidates), `maskbolt_fill_holes`, `maskbolt_export`. Find and view replies include a
numbered PNG; click replies include a zoom on the new mask. The server reads and writes only inside `--root`
(default: the folder it starts in; it refuses to start when that is a drive or file-system root or your home
folder). Paths are checked before any file access: UNC and device paths (`\\server\share`, `\\?\`, `\\.\`,
`//host`), other drives and anything `..` takes outside the root are refused with one message. Exports never
replace an existing file unless the call passes `overwrite: true`, and never replace the open image. View replies
give `image_x = region.x + preview_x / scale`; previews are at most 1568 px on the long side. On first use the model downloads
in the background; `maskbolt_open` says its size and asks to be called again if it is not done within 25 s.

The loop that works: open; find (add `tile: 400` for small or dense objects) or click; view (zoom with
`region`); remove wrong masks; click missed ones; gaps with `add: true` when replies list uncovered patches;
export.

## Speed (Ryzen 9 7950X3D, 16 cores, 64 GB, CPU only, release build, 2026-10-23)

| Step | SAM ViT-B | MobileSAM (`--model tiny`) |
|---|---|---|
| Load model | 0.2 s | 0.05-0.2 s |
| Encode one image (any size; SAM works at 1024 px) | 6.3-6.8 s | 4.2-4.3 s |
| One click (decode + upsample the three masks) | 0.11-0.12 s | 0.13-0.14 s |
| `cutout` on chelsea.png, whole command | 6.8 s, 2.0 GB peak | 5.1 s, 0.5 GB peak |
| `find` corn.png 1200x1198, 32x32 grid | 30.8 s wall (23.5 s after encode), 228 masks, 3.3 GB peak | 29.4 s wall, 234 masks, 3.0 GB peak |
| `find --tiles 400` (25 tiles, 16x16 each) | 221 s wall, 247 masks, 7.0 GB peak | not measured |
| `find --tiles 400 --fill-gaps` | 246 s wall, 256 masks | not measured |

Tiles encode 4 at a time (MASKBOLT_ENCODE_THREADS, default 4; each holds about 1.2 GB at its peak; 1 gives
about 3.3 GB peak and 332 s for the tiled corn run). Prompts decode on up to 16 threads (MASKBOLT_DECODE_THREADS).
For comparison, the Node prototype (8-bit ONNX) took 199 s for the 32x32 corn run and 416 s for the tiled one.

## Limits

- CPU only. Encoding is the slow step (about 6.5 s per image or tile with ViT-B); MobileSAM encodes in about
  4.2 s here, only 1.5x faster, because candle's CPU convolutions are slow. Memory peaks at 2-3.3 GB for one
  image and about 7 GB for tiled finds with 4 parallel encodes.
- Long tiled finds can outlast an MCP client's tool timeout (the 400 px corn run, 25 tiles, takes 3.5 minutes;
  the cap of 64 tiles is about 4 to 10 minutes). There are no progress notifications yet.
- SAM's masks miss thin parts and leave holes; use `level`, extra include/exclude points, and fill_holes.
- Outlines are simplified with Douglas-Peucker (1.2 px by default; `simplify_px` in MCP exports); edges move by
  up to that tolerance. Coordinates are image pixels from the top-left corner (pixel x covers [x, x+1]).
- The gap check is a colour heuristic: it finds look-alikes of what is already masked, not new kinds of object.

## Licence

MIT. Model weights: Apache-2.0 (Meta's SAM; MobileSAM). `src/sam_decoder.rs` is adapted from candle (MIT OR
Apache-2.0); candle's licence texts are in THIRD_PARTY.txt. Release archives and the MCPB bundle add
THIRD_PARTY_NOTICES.md with the licence texts of every Rust dependency in that build. Releasing: see
[mcpb/PUBLISHING.md](mcpb/PUBLISHING.md).
