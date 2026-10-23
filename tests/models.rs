//! Model cache (review P2-5): concurrent first use on one cache folder, the converted file's SHA-256 sidecar,
//! and re-conversion when the converted file no longer matches. Uses a tiny hand-made PyTorch checkpoint
//! (tests/data/tiny.pth, written by make_tiny_pth.py) with a strided tensor, an offset view and an int64 tensor.
//! This file is its own test binary because it sets MASKBOLT_MODEL_DIR.
use candle_core::Device;
use maskbolt::models::{self, ModelKind, ModelSpec};
use std::path::{Path, PathBuf};

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn values(p: &Path) -> std::collections::HashMap<String, Vec<f32>> {
    candle_core::safetensors::load(p, &Device::Cpu).unwrap().into_iter().map(|(k, t)| (k, t.flatten_all().unwrap().to_vec1::<f32>().unwrap())).collect()
}

#[test]
fn p2_5_model_cache_concurrent_first_use_and_verified_loads() {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("maskbolt-test-models-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("MASKBOLT_MODEL_DIR", &dir);
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny.pth");
    let sha = models::sha256_file(&fixture).unwrap();
    let bytes = std::fs::metadata(&fixture).unwrap().len();
    std::fs::copy(&fixture, dir.join("tiny-test.pth")).unwrap();
    let spec: &'static ModelSpec = Box::leak(Box::new(ModelSpec {
        kind: ModelKind::Tiny,
        name: "tiny-test",
        file: "tiny-test.pth",
        // Never reached: the checked .pth is already in the cache folder.
        url: "http://127.0.0.1:9/tiny-test.pth",
        sha256: leak(sha),
        bytes,
        licence: "test",
        project: "test",
    }));

    // Eight first uses at once on one cache folder.
    let handles: Vec<_> = (0..8).map(|_| std::thread::spawn(move || models::ensure(spec, |_, _| {}).map(|f| f.path).map_err(|e| format!("{e:#}")))).collect();
    let paths: Vec<Result<String, String>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for p in &paths {
        assert!(p.is_ok(), "a concurrent first use failed: {paths:?}");
    }
    let out = PathBuf::from(paths[0].as_ref().unwrap());
    assert!(paths.iter().all(|p| p.as_ref().unwrap() == paths[0].as_ref().unwrap()));
    let sidecar = PathBuf::from(format!("{}.sha256", out.display()));
    let recorded = std::fs::read_to_string(&sidecar).expect("a SHA-256 sidecar next to the converted file");
    assert_eq!(recorded.trim(), models::sha256_file(&out).unwrap());

    let v = values(&out);
    assert_eq!(v["plain"], vec![0., 1., 2., 3., 4., 5.]);
    assert_eq!(v["strided"], vec![0., 3., 1., 4., 2., 5.], "strided tensors are copied element by element");
    assert_eq!(v["offset"], vec![4., 5.], "candle's byte offset is converted to elements");
    assert!(!v.contains_key("counter"), "integer bookkeeping tensors are skipped");

    // A converted file that no longer matches its sidecar is rebuilt from the checked .pth.
    std::fs::write(&out, b"not a safetensors file").unwrap();
    let again = models::ensure(spec, |_, _| {}).unwrap();
    assert_eq!(PathBuf::from(&again.path), out);
    assert_eq!(std::fs::read_to_string(&sidecar).unwrap().trim(), models::sha256_file(&out).unwrap());
    assert_eq!(values(&out)["strided"], vec![0., 3., 1., 4., 2., 5.]);

    // No temp files are left behind.
    let leftovers: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).filter(|n| n.contains(".tmp") || n.contains(".part")).collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    std::fs::remove_dir_all(&dir).ok();
}
