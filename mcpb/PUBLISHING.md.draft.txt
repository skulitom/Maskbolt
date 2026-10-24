# Publishing Maskbolt

Pushing a version tag runs **Release**: locked builds with Rust 1.98.1 for Windows x64 (the C
runtime linked statically through `.cargo/config.toml`, so no Visual C++ Redistributable is
needed; the workflow fails if `maskbolt.exe` still imports `VCRUNTIME140`), Linux x64 (Ubuntu
22.04, glibc 2.35), macOS Apple silicon and macOS Intel, each on its own native runner. Each build
generates the dependency licence notices for its target, runs `maskbolt version` and starts the
MCP server (initialize, every catalog page, three calls that need no model), then packages the
native archive. The Windows build also builds the MCPB bundle, verifies it from a fresh extraction,
runs the packaging tests, validates the manifest with MCPB 2.1.2 and writes the registry entry,
which a separate job validates with checksum-pinned mcp-publisher 1.8.1. After every build
succeeds, the release job writes `SHA256SUMS`, attests the archives and bundle, and creates the
GitHub release. `maskbolt-mcpb` keeps the bundle, its checksum and `server.registry.json` as a
workflow artifact. Reruns skip identical release assets and fail on different bytes; they never
replace assets. Only the highest stable tag becomes Latest.

Pull requests touching the release workflows, `.cargo/`, `Cargo.lock`, `mcpb/` or `server.json`
run every build and check as a dry run, without attestation or release. Registry publication is
always a separate manual dispatch after testing. It uses GitHub OIDC (no account or stored secret)
and skips versions already listed.

**Published registry versions are immutable.** The registry workflow publishes only an installable
entry: the release's MCPB with its SHA-256. It refuses a release without the bundle; there is no
metadata-only fallback.

## Release

Bump `[package] version` in Cargo.toml (and `server.json`, the plugin's version in
`.claude-plugin/plugin.json`, the skill's install lines and the CHANGELOG) in a reviewed PR, merge
it, then tag the merge commit and push the tag:

```sh
git tag vX.Y.Z <merge commit>
git push origin vX.Y.Z
```

The tag must be `v` plus Cargo.toml's `[package] version`, a stable `X.Y.Z`. A manual Release run
also requires an existing tag and must start from it (`gh workflow run release.yml --ref vX.Y.Z -f
tag=vX.Y.Z`). Never reuse a tag for changed bytes. The manifest template has no version: the
builder inserts Cargo's version. `server.json` stays metadata-only; the entry writer stamps the
release version into its generated copy, checks the bundle's version and SHA-256, and never edits
the source metadata.

The builder reads the binary's live tool catalog and logs added or dropped names;
`mcpb/manifest.json` keeps curated one-line descriptions, not a fixed tool inventory. New tools use
the first sentence of the server's description, shortened to at most 160 characters on a word
boundary.

## Test the download, then publish

On Windows, use a checkout whose Cargo.toml matches the release, with Node 20 or newer:

```sh
gh release download vX.Y.Z -R skulitom/Maskbolt -p maskbolt-windows-x64.mcpb -p maskbolt-windows-x64.mcpb.sha256 -D target/release-test
gh attestation verify target/release-test/maskbolt-windows-x64.mcpb --repo skulitom/Maskbolt --signer-workflow skulitom/Maskbolt/.github/workflows/release.yml
node mcpb/verify.mjs target/release-test/maskbolt-windows-x64.mcpb
```

Compare `Get-FileHash target/release-test/maskbolt-windows-x64.mcpb -Algorithm SHA256` with the
downloaded `.sha256`. The verifier extracts to fresh folders under `target/`, launches the
manifest command with a temporary root folder and a stand-in home and model folder, checks every
catalog page and runs the model-free checks (a path outside the root, a missing image, a view with
no image open). It checks for unexpected writes in the temporary folders and changes elsewhere in
the checkout, then stops the server and removes its folders. This is a regression check, not an OS
sandbox. Non-Windows runs fail unless given `--allow-skip`; a skip is not verification.

Then test segmentation with the real model: extract the Windows ZIP, run `maskbolt version`, and
over MCP open a test image in a temporary root and click one object. Then dispatch:

```sh
gh workflow run publish-registry.yml -R skulitom/Maskbolt -f tag=vX.Y.Z
```

Publication requires the release and its bundle, verifies the bundle's attestation from the
Release workflow, validates the generated entry with mcp-publisher 1.8.1 and reads back the
published `fileSha256`. Tagged Cargo.toml and server.json supply the metadata; the publishing
helpers come from the workflow commit. Read the step summary and the
[registry listing](https://registry.modelcontextprotocol.io/v0/servers?search=io.github.skulitom/maskbolt).

## Local distribution checks

```sh
cargo build --release --locked
node mcpb/notices.mjs --target x86_64-pc-windows-msvc --out target/THIRD_PARTY_NOTICES.md
node mcpb/archive.mjs --target x86_64-pc-windows-msvc --binary target/release/maskbolt.exe --notices target/THIRD_PARTY_NOTICES.md
node mcpb/build.mjs --binary target/release/maskbolt.exe --notices target/THIRD_PARTY_NOTICES.md
node --test mcpb/mcpb.test.mjs
node mcpb/verify.mjs target/dist/mcpb/maskbolt-windows-x64.mcpb
node mcpb/build.mjs --registry-entry --tag vX.Y.Z --bundle target/dist/mcpb/maskbolt-windows-x64.mcpb
```

Run them on their own, without other builds or edits in the same checkout: the bundle verifier
inventories the checkout to detect unexpected writes. Outputs default to `target/dist/`, which Git
ignores, and are create-only: equal bytes are skipped, different bytes need a fresh `--out`. ZIP
entry order, timestamps and text line endings are fixed, so the same binary and Node version give
byte-identical bundles.

The bundle's root folder has no default: Maskbolt refuses to serve a drive root or the home
folder, so the user picks an image or project folder at install. Both distributions carry
`LICENSE`, `THIRD_PARTY.txt` and the generated `THIRD_PARTY_NOTICES.md`; native archives also
carry README.md. The model weights are not bundled: they download on first use from Meta's and
MobileSAM's official URLs, SHA-256 checked.

Listings outside the MCP Registry (the Anthropic plugin directory, Glama, awesome-mcp-servers) are
manual.
