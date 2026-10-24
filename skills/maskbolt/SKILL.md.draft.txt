---
name: maskbolt
description: "Cut objects out of images and outline them with Segment Anything (SAM), on this computer's CPU: remove the background behind one object as a transparent PNG, find and count every object in a photo, and export masks, numbered overlays, SVG or JSON outlines (with holes, optionally in millimetres) and label images. Use when someone asks to cut out, isolate or mask an object, remove a background, segment or count the objects in an image, or trace shapes in a photo into vector outlines (for sprites, stickers, product shots, laser or CNC offcuts). Local: no GPU, no Python, no account, nothing uploaded."
license: MIT
compatibility: "Needs the maskbolt executable on PATH, from the Maskbolt v0.1.0 release (Windows x64, Linux x64, macOS Apple silicon and Intel) or cargo install. The first image downloads the SAM model once (375 MB)."
---

# Maskbolt: Segment Anything on the CPU

Maskbolt runs Meta's Segment Anything (SAM ViT-B) on this computer's CPU. It segments one object
from a click, or every object in an image, shows numbered previews you can look at, and writes
cutouts and outlines. Nothing is uploaded and no GPU is needed.

## When to use it, and when not

| Request | Use |
| --- | --- |
| Cut the mug out of this photo / remove the background behind it | `maskbolt_open`, `maskbolt_click`, then `maskbolt_export` with `format: "cutout"` |
| How many coins, kernels or parts are in this picture? | `maskbolt_open`, `maskbolt_find`, check with `maskbolt_view`, then count |
| Outline every offcut on the bed as SVG in millimetres | `maskbolt_find`, fix the masks, then `maskbolt_export` with `format: "svg"` and `mm_per_px` |
| Masks for a game sprite, a sticker or a product shot | `maskbolt_click` per object, `maskbolt_export` `cutout` or `labels` |
| Resize, crop, convert, colour-correct or draw on an image | Not Maskbolt: it only segments; use an image editor or a script |
| Name or describe what is in the image | Not Maskbolt: SAM finds shapes, not labels; look at the image yourself |
| Video, or segmenting by a text prompt ("all the dogs") | Not in v0.1.0: one still image at a time, prompts are points and boxes |

## Paths and files

Every path is inside the server's root: the folder Claude Code was started in. Relative paths
are resolved against it; `..`, other drives and network paths are refused. Images are PNG, JPEG
or WebP up to 100 megapixels. Exports never replace an existing file unless the call passes
`overwrite: true`, and never the open image; missing output folders are created.

Coordinates are image pixels from the top-left corner. Previews can be scaled: a preview pixel
maps back as `image_x = region.x + preview_x / scale` (the same for y), using the `region` and
`scale` in the reply.

## The loop that works

1. **Open.** `maskbolt_open` with `path`. It encodes the image (about 7 s on a 16-core CPU,
   longer on a laptop) and clears all masks. On first use it downloads the model (375 MB,
   SHA-256 checked); if the reply says `"status": "downloading model"`, call `maskbolt_open`
   again in a minute.
2. **Find or click.**
   - One object: `maskbolt_click` with `points: [[x, y]]` on it. Add points with a third value
     `0` (or `exclude: [[x, y]]`) for things that are not part of it, and a `box: [x0, y0, x1, y1]`
     to hold it in. The reply lists all three SAM sizes (`smallest` a part, `middle` the object,
     `largest` the object with what it holds) and keeps `best` by default.
   - Every object: `maskbolt_find`. For small or dense objects (kernels, screws, coins in a big
     photo) add `tile: 400`; a tiled find drops objects larger than about a third of a tile, so
     run a plain find first when the image also has big objects. Tiled finds take minutes.
3. **View.** Look at the numbered preview in each reply, and call `maskbolt_view` with a
   `region` to zoom on small or crowded objects. Check every object has exactly one mask.
4. **Remove** wrong, merged or duplicate masks by id with `maskbolt_remove`. To switch an
   object to another level, remove its mask and click again with `level`; a click that
   duplicates, lies inside or contains an existing mask is skipped unless `force: true`.
5. **Click missed ones** with `maskbolt_click`.
6. **Gaps.** When a reply's `still_uncovered` lists patches, call `maskbolt_gaps` (it lists
   look-alike uncovered patches with a click point each) or `maskbolt_gaps` with `add: true` to
   click them all. It finds more of what is already masked, not new kinds of object.
   `maskbolt_fill_holes` fills holes SAM left inside a mask (not where holes are real, such as
   rings or washers).
7. **Export** when the masks are right, with `maskbolt_export`: `cutout` (transparent PNG of the
   chosen `ids`), `overlay` (numbered PNG), `svg` (outer loops and holes, even-odd; `mm_per_px`
   for millimetres), `json` (outlines with holes, area, box, score) or `labels` (one colour per
   mask). Tell the user the file paths, how many masks each holds, and what you could not fix.

SAM misses thin parts and leaves holes: use `level`, extra include and exclude points, a box,
and `fill_holes` rather than accepting a poor mask. Outlines are simplified by 1.2 px by default
(`simplify_px`, 0 to 20).

## If Maskbolt's tools are missing

The Claude Code plugin registers the `maskbolt` server and this skill, but it doesn't install the
program. Without `maskbolt` on PATH the server fails to start (`/mcp` shows
`plugin:maskbolt:maskbolt` failed) and no Maskbolt tools appear. It also refuses to start when
Claude Code runs in a drive root or the home folder: start it in a project or image folder. With
the user's agreement:

1. Download the archive for their computer from the
   [v0.1.0 release](https://github.com/skulitom/Maskbolt/releases/tag/v0.1.0):
   `maskbolt-v0.1.0-x86_64-pc-windows-msvc.zip`, `maskbolt-v0.1.0-x86_64-unknown-linux-gnu.tar.gz`,
   `maskbolt-v0.1.0-aarch64-apple-darwin.tar.gz` or `maskbolt-v0.1.0-x86_64-apple-darwin.tar.gz`,
   and check it against `SHA256SUMS` from the same release. With Rust 1.89 or newer,
   `cargo install --git https://github.com/skulitom/Maskbolt --tag v0.1.0 --locked` builds it instead.
2. Put `maskbolt` (`maskbolt.exe` on Windows) in a folder on the user's PATH; `maskbolt version`
   should print `maskbolt 0.1.0`.
3. Optionally run `maskbolt fetch` to download the model now (375 MB into
   `%LOCALAPPDATA%\maskbolt\models`, or `~/.cache/maskbolt/models`).
4. Restart Claude Code from a new terminal, so the plugin finds it.

More: the [README](https://github.com/skulitom/Maskbolt#readme).
