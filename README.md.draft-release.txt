Proposed README.md changes for the 0.1.0 release: only the sections that change. Replace each OLD block with
its NEW block; everything else in README.md stays as it is. Check the commands against the published release
before merging (asset names come from .github/workflows/release.yml).

=====================================================================================================
1. "## Install": replace the first code line

OLD:
    cargo install --path .        (crates.io and release binaries: not published yet)

NEW:
Download the archive for your computer from the [latest release](https://github.com/skulitom/Maskbolt/releases/latest):
`maskbolt-v0.1.0-x86_64-pc-windows-msvc.zip`, `maskbolt-v0.1.0-x86_64-unknown-linux-gnu.tar.gz` (glibc 2.35 or
newer), `maskbolt-v0.1.0-aarch64-apple-darwin.tar.gz` or `maskbolt-v0.1.0-x86_64-apple-darwin.tar.gz`. Check it
against the release's `SHA256SUMS`, extract it and put `maskbolt` (`maskbolt.exe` on Windows) on your PATH;
`maskbolt version` prints `maskbolt 0.1.0`. Each archive also holds the licence, THIRD_PARTY.txt and
THIRD_PARTY_NOTICES.md (the licence texts of the Rust dependencies in that build). Or build it with Rust 1.89 or
newer:

    cargo install --git https://github.com/skulitom/Maskbolt --tag v0.1.0 --locked

Maskbolt is not on crates.io.

(The paragraph that follows, "The first run downloads the model into a user cache folder ...", and the model
table stay.)

=====================================================================================================
2. "## MCP": after the JSON line
    {"mcpServers": {"maskbolt": {"command": "maskbolt", "args": ["mcp", "--root", "C:\\work\\images"]}}}
insert:

NEW:
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

(The paragraph "Tools: `maskbolt_open`, ..." and everything after it stay.)

=====================================================================================================
3. "## Licence": append one sentence

OLD:
MIT. Model weights: Apache-2.0 (Meta's SAM; MobileSAM). `src/sam_decoder.rs` is adapted from candle (MIT OR
Apache-2.0); candle's licence texts are in THIRD_PARTY.txt.

NEW:
MIT. Model weights: Apache-2.0 (Meta's SAM; MobileSAM). `src/sam_decoder.rs` is adapted from candle (MIT OR
Apache-2.0); candle's licence texts are in THIRD_PARTY.txt. Release archives and the MCPB bundle add
THIRD_PARTY_NOTICES.md with the licence texts of every Rust dependency in that build. Releasing: see
[mcpb/PUBLISHING.md](mcpb/PUBLISHING.md).
