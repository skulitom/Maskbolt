//! Regression tests for the 2026-10-24 review (REVIEW.txt): one per finding, each seen failing before its fix.
mod common;
use common::*;
use maskbolt::export::{self, ExportArgs, Format};
use maskbolt::mcp::Maskbolt;
use maskbolt::models::ModelKind;
use maskbolt::paths::Root;
use maskbolt::{ClickArgs, FakeBackend, FindArgs, Mask, Point, Region, Session};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn session(img: image::RgbImage) -> Session {
    let mut s = Session::new(Arc::new(FakeBackend::default()));
    s.open_rgb(img, PathBuf::from("synthetic.png")).unwrap();
    s
}

/// Run `f` on another thread; fail the test if it does not finish in `secs`.
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs)).unwrap_or_else(|_| panic!("did not finish within {secs} s"))
}

// P1-1: label colours probed only the blue byte and looped for ever past about 1,550 masks.
#[test]
fn p1_label_colours_are_distinct_for_5000_masks_and_fast() {
    let mut s = session(canvas(200, 200, WHITE));
    for i in 0..5000u32 {
        let (x, y) = (i % 100 * 2, i / 100 * 2);
        s.add(Mask::from_fn(x, y, 1, 1, |_, _| true), 0.9, "t", true).unwrap();
    }
    let dir = temp_dir("p1");
    let (colours, json_count) = within(20, move || {
        let t = Instant::now();
        let c = export::label_colors(&s);
        let mut a = ExportArgs::new(Format::Labels);
        a.ids = None;
        export::export(&s, &a, &dir.join("labels.png")).unwrap();
        export::export(&s, &ExportArgs::new(Format::Json), &dir.join("masks.json")).unwrap();
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("masks.json")).unwrap()).unwrap();
        assert!(t.elapsed() < Duration::from_secs(10));
        std::fs::remove_dir_all(&dir).ok();
        (c, j["count"].as_u64().unwrap())
    });
    assert_eq!(colours.len(), 5000);
    let unique: std::collections::HashSet<[u8; 3]> = colours.values().copied().collect();
    assert_eq!(unique.len(), 5000, "one colour per mask");
    assert!(colours.values().all(|c| c.iter().copied().max().unwrap() >= 48), "no near-black label colours");
    assert_eq!(json_count, 5000);
}

// P2-1: UNC and other outside paths must be refused before any file-system call, with one message whether or
// not the file exists.
#[test]
fn p2_1_unc_and_outside_paths_are_refused_lexically() {
    let base = temp_dir("p21");
    let root_dir = base.join("root");
    std::fs::create_dir_all(&root_dir).unwrap();
    std::fs::write(base.join("exists.png"), b"x").unwrap();
    let root = Root::new(&root_dir).unwrap();
    // 192.0.2.1 is TEST-NET-1: any SMB attempt would take seconds before failing.
    let unc = [r"\\192.0.2.1\share\a.png", "//192.0.2.1/share/a.png", r"\\?\C:\Windows\win.ini", r"\\.\C:\Windows\win.ini", r"\/192.0.2.1\share\a.png"];
    let verbatim_inside = format!(r"\\?\{}\a.png", root.dir().display());
    for p in unc.iter().map(|s| s.to_string()).chain([verbatim_inside]) {
        let t = Instant::now();
        let r = root.read(&p).unwrap_err().to_string();
        let w = root.write(&p).unwrap_err().to_string();
        assert!(t.elapsed() < Duration::from_millis(200), "{p}: took {:?} (touched the network?)", t.elapsed());
        assert!(r.contains("outside the allowed root") && w.contains("outside the allowed root"), "{p}: {r} / {w}");
    }
    // Outside files: the same text whether or not they exist.
    let exists = base.join("exists.png").display().to_string();
    let missing = base.join("missing.png").display().to_string();
    let e1 = root.read(&exists).unwrap_err().to_string().replace(&exists, "<p>");
    let e2 = root.read(&missing).unwrap_err().to_string().replace(&missing, "<p>");
    assert_eq!(e1, e2, "existence must not leak");
    // Other drives, drive-relative paths, alternate data streams and Win32 trailing-dot names are refused.
    if cfg!(windows) {
        for p in [r"Z:\x.png", "C:x.png", "a.png:stream", "sub\\.. \\x.png", "x.png."] {
            assert!(root.write(p).is_err(), "{p} was accepted");
        }
    }
    // Plain paths inside still work.
    assert!(root.write("out/ok.png").is_ok());
    std::fs::remove_dir_all(&base).ok();
}

// P2-2: the MCP server must not serve a drive root or the home folder.
#[test]
fn p2_2_root_cannot_be_a_drive_root_or_home() {
    let tmp = temp_dir("p22");
    let drive_root: PathBuf = tmp.ancestors().last().unwrap().to_path_buf();
    let e = Maskbolt::new(&drive_root, ModelKind::VitB).err().expect("a drive root is refused").to_string();
    assert!(e.contains("--root"), "{e}");
    if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
        let e = Maskbolt::new(Path::new(&home), ModelKind::VitB).err().expect("the home folder is refused").to_string();
        assert!(e.contains("--root"), "{e}");
    }
    assert!(Maskbolt::new(&tmp, ModelKind::VitB).is_ok());
    std::fs::remove_dir_all(&tmp).ok();
}

// P2-3: exports must not replace existing files by default, nor ever the open image.
#[test]
fn p2_3_export_does_not_overwrite_by_default() {
    let dir = temp_dir("p23");
    let img_path = dir.join("scene.png");
    ring_scene().save(&img_path).unwrap();
    let original = std::fs::read(&img_path).unwrap();
    let mut s = Session::new(Arc::new(FakeBackend::default()));
    s.open(&img_path).unwrap();
    s.click(&ClickArgs { points: vec![Point { x: 50.0, y: 80.0, include: true }], ..Default::default() }).unwrap();
    // The open image, named another way.
    let other_name = dir.join(".").join("scene.png");
    let e = export::export(&s, &ExportArgs::new(Format::Overlay), &other_name).unwrap_err().to_string();
    assert!(e.contains("open image"), "{e}");
    assert_eq!(std::fs::read(&img_path).unwrap(), original, "the open image was replaced");
    // Any other existing file.
    std::fs::write(dir.join("keep.json"), b"mine").unwrap();
    let e = export::export(&s, &ExportArgs::new(Format::Json), &dir.join("keep.json")).unwrap_err().to_string();
    assert!(e.contains("already exists"), "{e}");
    assert_eq!(std::fs::read(dir.join("keep.json")).unwrap(), b"mine");
    std::fs::remove_dir_all(&dir).ok();
}

// P2-3 (new option): overwrite replaces other files, but never the open image.
#[test]
fn p2_3_overwrite_flag_never_touches_the_open_image() {
    let dir = temp_dir("p23b");
    let img_path = dir.join("scene.png");
    ring_scene().save(&img_path).unwrap();
    let mut s = Session::new(Arc::new(FakeBackend::default()));
    s.open(&img_path).unwrap();
    s.click(&ClickArgs { points: vec![Point { x: 50.0, y: 80.0, include: true }], ..Default::default() }).unwrap();
    let mut a = ExportArgs::new(Format::Overlay);
    a.overwrite = true;
    assert!(export::export(&s, &a, &img_path).is_err());
    std::fs::write(dir.join("old.png"), b"old").unwrap();
    export::export(&s, &a, &dir.join("old.png")).unwrap();
    assert_ne!(std::fs::read(dir.join("old.png")).unwrap(), b"old");
    std::fs::remove_dir_all(&dir).ok();
}

// P2-4: one find is capped at 64 tiles and a 48x48 grid, with an estimate in the error.
#[test]
fn p2_4_find_caps_tiles_and_grid() {
    let mut s = session(canvas(2000, 1500, WHITE));
    let t = Instant::now();
    let e = s.find(&FindArgs { grid: 1, tile: Some(32), ..Default::default() }).unwrap_err().to_string();
    assert!(e.contains("tiles") && e.contains("64") && e.contains("region"), "{e}");
    assert!(e.contains("min"), "a time estimate: {e}");
    assert!(t.elapsed() < Duration::from_secs(2), "refused before encoding anything");
    assert!(s.find(&FindArgs { grid: 49, ..Default::default() }).is_err());
    let mut small = session(canvas(64, 64, WHITE));
    assert!(small.find(&FindArgs { grid: 48, ..Default::default() }).is_ok());
    // 64 tiles is allowed: 8 x 8 tiles of 250 px with stride 150 cover 1200 x 1200.
    let mut s = session(canvas(1200, 1200, WHITE));
    assert_eq!(s.find(&FindArgs { grid: 1, tile: Some(250), ..Default::default() }).unwrap().crops, 64);
}

// P2-6: the adapted candle code carries candle's licence texts.
#[test]
fn p2_6_candle_licence_texts_ship_with_the_adapted_code() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let third = std::fs::read_to_string(root.join("THIRD_PARTY.txt")).unwrap();
    assert!(third.contains("Permission is hereby granted, free of charge"), "candle's MIT text");
    assert!(third.contains("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION"), "the Apache-2.0 text");
    assert!(!third.contains("Licence texts ship inside each crate's source"));
    let header: String = std::fs::read_to_string(root.join("src/sam_decoder.rs")).unwrap().lines().take(12).collect::<Vec<_>>().join("\n");
    assert!(header.contains("huggingface/candle") && header.contains("MIT OR Apache-2.0") && header.contains("THIRD_PARTY.txt"), "{header}");
}

// P3: Region::overlaps overflowed (debug builds panicked) for x near u32::MAX.
#[test]
fn p3_region_far_out_does_not_overflow() {
    let mut s = session(ring_scene());
    s.click(&ClickArgs { points: vec![Point { x: 50.0, y: 80.0, include: true }], ..Default::default() }).unwrap();
    assert_eq!(s.summary(Some(Region { x: 100, y: 0, w: u32::MAX, h: u32::MAX }), None).len(), 1);
}

// P3: the decoder was chosen by file extension, so a PNG named .jpg failed.
#[test]
fn p3_images_open_by_content_not_extension() {
    let dir = temp_dir("p3img");
    let p = dir.join("really-a-png.jpg");
    ring_scene().save_with_format(&p, image::ImageFormat::Png).unwrap();
    let mut s = Session::new(Arc::new(FakeBackend::default()));
    assert_eq!(s.open(&p).unwrap().width, 200);
    std::fs::remove_dir_all(&dir).ok();
}
