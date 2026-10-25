# Privacy

Maskbolt (the `maskbolt` CLI, its MCP server, the Claude Code plugin and the MCPB bundle) runs entirely on your
computer. It has no accounts, no API keys, no telemetry, no analytics and no update checks, and it never uploads
your images or masks. Its only network request is the one-time model download described below.

## What it reads

- **Only the images you or your agent name.** The MCP server reads and writes only inside one root folder: the
  `--root` it is started with, which the Claude Code plugin sets to your project folder, the MCPB bundle sets to
  the image folder you choose, and which otherwise is the folder it was started in. Paths outside the root are
  refused: other drives, UNC and device paths, `..` that leaves the root, and links or junctions that resolve
  outside it. A drive root or your home folder is refused as a root. Folder checks are not an operating-system
  sandbox. (`src/paths.rs`)
- The CLI reads the one image path you give it.
- Images stay in memory while you work on them; Maskbolt keeps no copies, history or logs. Progress messages go to
  standard error.

## What it writes

- **Exports, where you ask:** cutout PNGs, numbered overlay PNGs, SVG and JSON outlines and label PNGs, written to
  the path in the request (inside the root, for the MCP server). An existing file is replaced only when the request
  says so (`overwrite: true`, or `--force` on the CLI), the image you opened is never replaced, and the MCP server
  does not write through links. (`src/export.rs`, `src/paths.rs`, `src/main.rs`)
- **The model weights**, in a cache folder: `MASKBOLT_MODEL_DIR` if set (the Docker image uses `/cache`), else
  `%LOCALAPPDATA%\maskbolt\models` on Windows, else `$XDG_CACHE_HOME/maskbolt/models`, else
  `~/.cache/maskbolt/models`. It holds the downloaded checkpoint, the converted `.safetensors` file and a
  `.sha256` file recording that file's hash. Delete the folder to remove them. (`src/models.rs`)

## Network

The first time a model is used (or when you run `maskbolt fetch`), Maskbolt downloads its official weights over
HTTPS from the one URL pinned in `src/models.rs`:

| Model | Host | URL |
|---|---|---|
| SAM ViT-B (default, 375 MB, Meta) | `dl.fbaipublicfiles.com` | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` |
| MobileSAM (`--model tiny`, 41 MB) | `github.com` (GitHub serves the file from `raw.githubusercontent.com`) | `https://github.com/ChaoningZhang/MobileSAM/raw/a509aac54fdd7af59f843135f2f7cee307283c88/weights/mobile_sam.pt` |

The download is checked against the SHA-256 and exact size pinned in the code while it streams, and deleted if
either differs. The request carries nothing about you or your images; as with any download, the host sees your IP
address and a generic `ureq/<version>` user agent. After that Maskbolt works offline. There is no other network
code: the download in `src/models.rs` is the only use of an HTTP client, and the MCP server listens on no port.

## What your MCP client sees

The MCP server talks only to the client that started it, over standard input and output. Its replies (JSON with
mask counts, boxes, outlines and file paths, and PNG previews of your image with the masks drawn on it) go to that
client, for example Claude Code, which passes them to its AI model as part of the conversation. How the client and
its model provider handle that conversation is set by their privacy terms, not Maskbolt's.

## Contact

Questions or concerns: open an issue at <https://github.com/skulitom/Maskbolt/issues>.

Last updated 25 October 2026, for Maskbolt 0.1.0.
