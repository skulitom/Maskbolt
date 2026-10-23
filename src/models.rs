//! Model weights: official checkpoints downloaded on first use into a user cache folder, checked with SHA-256,
//! then converted once from PyTorch .pth to safetensors (candle cannot read some strided tensors in Meta's file).
use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    VitB,
    Tiny,
}

impl std::str::FromStr for ModelKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "vit-b" | "vitb" | "vit_b" | "sam" | "default" | "base" => ModelKind::VitB,
            "tiny" | "mobilesam" | "mobile-sam" | "mobile_sam" => ModelKind::Tiny,
            _ => bail!("model must be vit-b (default) or tiny"),
        })
    }
}

pub struct ModelSpec {
    pub kind: ModelKind,
    pub name: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    pub licence: &'static str,
    pub project: &'static str,
}

pub const VIT_B: ModelSpec = ModelSpec {
    kind: ModelKind::VitB,
    name: "sam-vit-b",
    file: "sam_vit_b_01ec64.pth",
    url: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    sha256: "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    bytes: 375_042_383,
    licence: "Apache-2.0",
    project: "Segment Anything by Meta AI, https://github.com/facebookresearch/segment-anything",
};

pub const TINY: ModelSpec = ModelSpec {
    kind: ModelKind::Tiny,
    name: "mobile-sam",
    file: "mobile_sam.pt",
    url: "https://github.com/ChaoningZhang/MobileSAM/raw/a509aac54fdd7af59f843135f2f7cee307283c88/weights/mobile_sam.pt",
    sha256: "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    bytes: 40_728_226,
    licence: "Apache-2.0",
    project: "MobileSAM (TinyViT-5M encoder) by Chaoning Zhang et al., https://github.com/ChaoningZhang/MobileSAM",
};

pub fn spec(kind: ModelKind) -> &'static ModelSpec {
    match kind {
        ModelKind::VitB => &VIT_B,
        ModelKind::Tiny => &TINY,
    }
}

/// MASKBOLT_MODEL_DIR, else %LOCALAPPDATA%\maskbolt\models on Windows, else $XDG_CACHE_HOME/maskbolt/models,
/// else ~/.cache/maskbolt/models.
pub fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("MASKBOLT_MODEL_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    if cfg!(windows) {
        if let Some(d) = std::env::var_os("LOCALAPPDATA").filter(|d| !d.is_empty()) {
            return PathBuf::from(d).join("maskbolt").join("models");
        }
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME").filter(|d| !d.is_empty()) {
        return PathBuf::from(d).join("maskbolt").join("models");
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("maskbolt").join("models")
}

/// The converted weights used at run time.
pub fn converted_path(s: &ModelSpec) -> PathBuf {
    cache_dir().join(format!("{}.safetensors", s.name))
}

/// The SHA-256 of the converted file, written after conversion and checked on every load.
pub fn sidecar_path(s: &ModelSpec) -> PathBuf {
    PathBuf::from(format!("{}.sha256", converted_path(s).display()))
}

/// Cheap check (no hashing): the converted file and its sidecar exist. `ensure` verifies the hash.
pub fn is_ready(s: &ModelSpec) -> bool {
    converted_path(s).is_file() && sidecar_path(s).is_file()
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Fetched {
    pub path: String,
    pub downloaded: bool,
    pub bytes: u64,
    pub url: String,
    pub sha256: String,
    pub licence: String,
}

/// A temp name no other process or thread will pick: process id plus a random suffix.
fn unique(dir: &Path, base: &str) -> PathBuf {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    dir.join(format!("{base}.{}-{:016x}.tmp", std::process::id(), h.finish()))
}

fn converted_ok(out: &Path, side: &Path) -> bool {
    let Ok(want) = std::fs::read_to_string(side) else { return false };
    out.is_file() && sha256_file(out).is_ok_and(|got| got == want.trim())
}

/// Move `tmp` to `dest` atomically. If that fails because another process put `dest` in place first (and
/// `ok(dest)` says it is good), use theirs.
fn place(tmp: &Path, dest: &Path, ok: impl Fn() -> bool) -> Result<()> {
    for attempt in 0..10 {
        match std::fs::rename(tmp, dest) {
            Ok(()) => return Ok(()),
            Err(e) => {
                if ok() {
                    let _ = std::fs::remove_file(tmp);
                    return Ok(());
                }
                if attempt == 9 {
                    let _ = std::fs::remove_file(tmp);
                    return Err(e).with_context(|| format!("cannot move {} to {}", tmp.display(), dest.display()));
                }
                std::thread::sleep(std::time::Duration::from_millis(100 + 50 * attempt));
            }
        }
    }
    unreachable!()
}

/// Make sure verified, converted weights exist: download the checkpoint (SHA-256 and size checked while
/// streaming) if it is missing, convert it to safetensors, and record the converted file's SHA-256 in a sidecar.
/// Every call re-hashes the converted file against its sidecar and rebuilds it from the checked .pth on a
/// mismatch. Safe to run from several processes at once: temp files have unique names and land by rename.
/// `progress(done, total)` is called while downloading.
pub fn ensure(s: &ModelSpec, mut progress: impl FnMut(u64, u64)) -> Result<Fetched> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create model folder {}", dir.display()))?;
    let out = converted_path(s);
    let side = sidecar_path(s);
    let info = |downloaded| Fetched { path: out.display().to_string(), downloaded, bytes: s.bytes, url: s.url.into(), sha256: s.sha256.into(), licence: s.licence.into() };
    if converted_ok(&out, &side) {
        return Ok(info(false));
    }
    let src = dir.join(s.file);
    let src_ok = || src.is_file() && sha256_file(&src).is_ok_and(|h| h == s.sha256);
    let mut downloaded = false;
    if !src_ok() {
        let part = unique(&dir, s.file);
        download(s, &part, &mut progress)?;
        place(&part, &src, src_ok)?;
        downloaded = true;
    }
    for _ in 0..3 {
        if converted_ok(&out, &side) {
            return Ok(info(downloaded));
        }
        let tmp = unique(&dir, &format!("{}.safetensors", s.name));
        if let Err(e) = convert_pth(&src, &tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e).with_context(|| format!("converting {}", src.display()));
        }
        let sha = sha256_file(&tmp)?;
        place(&tmp, &out, || converted_ok(&out, &side))?;
        let side_tmp = unique(&dir, &format!("{}.sha256", s.name));
        std::fs::write(&side_tmp, &sha)?;
        place(&side_tmp, &side, || converted_ok(&out, &side))?;
    }
    if converted_ok(&out, &side) {
        Ok(info(downloaded))
    } else {
        bail!("the converted model {} does not match its recorded SHA-256 after three attempts", out.display())
    }
}

fn download(s: &ModelSpec, part: &Path, progress: &mut impl FnMut(u64, u64)) -> Result<()> {
    let agent = ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_connect(Some(std::time::Duration::from_secs(30)))
            .timeout_recv_response(Some(std::time::Duration::from_secs(60)))
            .timeout_recv_body(Some(std::time::Duration::from_secs(3600)))
            .build(),
    );
    let mut resp = agent.get(s.url).call().with_context(|| format!("downloading {}", s.url))?;
    let mut reader = resp.body_mut().as_reader();
    let mut file = std::fs::File::create(part)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    let result = (|| -> Result<()> {
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            done += n as u64;
            ensure!(done <= s.bytes, "{} is larger than the expected {} bytes", s.url, s.bytes);
            hasher.update(&buf[..n]);
            file.write_all(&buf[..n])?;
            progress(done, s.bytes);
        }
        file.sync_all()?;
        Ok(())
    })();
    drop(file);
    if let Err(e) = result {
        let _ = std::fs::remove_file(part);
        return Err(e);
    }
    let got = hex(&hasher.finalize());
    if got != s.sha256 || done != s.bytes {
        let _ = std::fs::remove_file(part);
        bail!("download of {} failed its check: got {done} bytes with SHA-256 {got}, expected {} bytes with {}", s.url, s.bytes, s.sha256);
    }
    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn sha256_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Read every tensor of a PyTorch zip checkpoint, copying strided (non-contiguous) tensors element by element,
/// and save them as safetensors.
pub fn convert_pth(src: &Path, dest: &Path) -> Result<()> {
    let infos = candle_core::pickle::read_pth_tensor_info(src, false, None)?;
    ensure!(!infos.is_empty(), "no tensors in {}", src.display());
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(std::fs::File::open(src)?))?;
    let mut out: HashMap<String, Tensor> = HashMap::new();
    for ti in infos {
        // Integer bookkeeping tensors (e.g. batch norm's num_batches_tracked) are not weights.
        if !matches!(ti.dtype, DType::F32 | DType::F16) {
            continue;
        }
        let mut raw = Vec::new();
        zip.by_name(&ti.path)?.read_to_end(&mut raw)?;
        let elem = ti.dtype.size_in_bytes();
        ensure!(raw.len() >= ti.storage_size * elem, "{}: storage is shorter than expected", ti.name);
        let storage: Vec<f32> = match ti.dtype {
            DType::F32 => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            DType::F16 => raw.as_chunks::<2>().0.iter().map(|c| half_to_f32(u16::from_le_bytes(*c))).collect(),
            other => bail!("{}: unsupported dtype {other:?}", ti.name),
        };
        let shape = ti.layout.shape().dims().to_vec();
        let stride = ti.layout.stride().to_vec();
        // candle records the storage offset in bytes.
        let start = ti.layout.start_offset() / elem;
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..n {
            let off = start + idx.iter().zip(&stride).map(|(i, s)| i * s).sum::<usize>();
            data.push(*storage.get(off).with_context(|| format!("{}: index {off} out of storage {} (raw {} bytes, layout {:?}, storage_size {})", ti.name, storage.len(), raw.len(), ti.layout, ti.storage_size))?);
            for d in (0..shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out.insert(ti.name.clone(), Tensor::from_vec(data, shape, &Device::Cpu)?);
    }
    candle_core::safetensors::save(&out, dest)?;
    Ok(())
}

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
