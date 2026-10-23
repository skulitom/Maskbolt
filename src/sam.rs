//! The candle backend: SAM ViT-B or MobileSAM (TinyViT) on CPU.
//! Image encoders come from candle-transformers; prompts are encoded here (so they batch, and boxes work);
//! the mask decoder is `sam_decoder`. Masks are SAM's 256x256 logits upsampled bilinearly to image pixels and
//! thresholded at 0, evaluated only around the positive cells (cheap for small objects).
use crate::backend::{Backend, Embedding, Prompt, Proposal};
use crate::mask::Mask;
use crate::models::ModelKind;
use crate::sam_decoder::MaskDecoder;
use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::{image_encoder::ImageEncoderViT, tiny_vit::{tiny_vit_5m, TinyViT}};
use image::RgbImage;
use std::path::Path;

pub const IMAGE_SIZE: usize = 1024;
const LOW: usize = 256;

enum Encoder {
    Vit(Box<ImageEncoderViT>),
    Tiny(Box<TinyViT>),
}

pub struct SamBackend {
    kind: ModelKind,
    encoder: Encoder,
    decoder: MaskDecoder,
    gauss: Tensor,
    point_embed: Vec<Tensor>,
    not_a_point: Tensor,
    dense: Tensor,
    image_pe: Tensor,
    dev: Device,
}

struct SamEmb {
    tensor: Tensor,
    input_w: usize,
    input_h: usize,
}

impl SamBackend {
    /// Load converted safetensors weights (see `models::ensure`).
    pub fn load(kind: ModelKind, path: &Path) -> Result<Self> {
        let dev = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &dev)? };
        let encoder = match kind {
            ModelKind::VitB => Encoder::Vit(Box::new(ImageEncoderViT::new(
                IMAGE_SIZE, 16, 3, 768, 12, 12, 256, true, true, true, 14, &[2, 5, 8, 11], vb.pp("image_encoder"),
            )?)),
            ModelKind::Tiny => Encoder::Tiny(Box::new(tiny_vit_5m(vb.pp("image_encoder"))?)),
        };
        let decoder = MaskDecoder::new(vb.pp("mask_decoder"))?;
        let pe = vb.pp("prompt_encoder");
        let gauss = pe.get((2, 128), "pe_layer.positional_encoding_gaussian_matrix")?;
        let point_embed = (0..4).map(|i| pe.get((1, 256), &format!("point_embeddings.{i}.weight"))).collect::<candle_core::Result<Vec<_>>>()?;
        let not_a_point = pe.get((1, 256), "not_a_point_embed.weight")?;
        let no_mask = pe.get((1, 256), "no_mask_embed.weight")?;
        let dense = no_mask.reshape((1, 256, 1, 1))?.broadcast_as((1, 256, 64, 64))?.contiguous()?;
        let n = 64usize;
        let mut coords = Vec::with_capacity(n * n * 2);
        for y in 0..n {
            for x in 0..n {
                coords.push((x as f32 + 0.5) / n as f32);
                coords.push((y as f32 + 0.5) / n as f32);
            }
        }
        let coords = Tensor::from_vec(coords, (n, n, 2), &dev)?;
        let image_pe = pe_encoding(&coords, &gauss)?.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
        Ok(Self { kind, encoder, decoder, gauss, point_embed, not_a_point, dense, image_pe, dev })
    }

    fn decode_job(&self, e: &SamEmb, emb: &Embedding, prompts: &[Prompt], idx: &[usize], sx: f32, sy: f32) -> Result<Vec<(usize, Proposal)>> {
        let ps: Vec<Prompt> = idx.iter().map(|&i| prompts[i].clone()).collect();
        let sparse = self.sparse(&ps, sx, sy)?;
        let (masks, iou) = self.decoder.forward(&e.tensor, &self.image_pe, &sparse, &self.dense)?;
        let masks = masks.flatten_all()?.to_vec1::<f32>()?;
        let iou = iou.flatten_all()?.to_vec1::<f32>()?;
        let per = LOW * LOW;
        let mut out = Vec::with_capacity(idx.len());
        for (k, &i) in idx.iter().enumerate() {
            let ms: Vec<Mask> = (0..3).map(|j| upsample(&masks[(k * 3 + j) * per..(k * 3 + j + 1) * per], e.input_w, e.input_h, emb.width, emb.height)).collect();
            let [a, b, c]: [Mask; 3] = ms.try_into().unwrap();
            out.push((i, Proposal { masks: [a, b, c], scores: [iou[k * 3], iou[k * 3 + 1], iou[k * 3 + 2]] }));
        }
        Ok(out)
    }

    fn sparse(&self, ps: &[Prompt], sx: f32, sy: f32) -> Result<Tensor> {
        let b = ps.len();
        let has_box = ps[0].bbox.is_some();
        let pad = !has_box;
        let k = ps[0].points.len() + usize::from(pad);
        let s = IMAGE_SIZE as f32;
        let mut parts = Vec::new();
        if k > 0 {
            let mut xy = Vec::with_capacity(b * k * 2);
            let mut lab = Vec::with_capacity(b * k);
            for p in ps {
                for q in &p.points {
                    xy.push((q.x * sx + 0.5) / s);
                    xy.push((q.y * sy + 0.5) / s);
                    lab.push(if q.include { 1f32 } else { 0f32 });
                }
                if pad {
                    xy.extend([0.0, 0.0]);
                    lab.push(-1.0);
                }
            }
            let xy = Tensor::from_vec(xy, (b, k, 2), &self.dev)?;
            let pe = pe_encoding(&xy, &self.gauss)?;
            let lab = Tensor::from_vec(lab, (b, k, 1), &self.dev)?.broadcast_as((b, k, 256))?;
            let zeros = pe.zeros_like()?;
            let shape = (b, k, 256);
            let nap = self.not_a_point.reshape((1, 1, 256))?.broadcast_as(shape)?;
            let e = lab.lt(0f32)?.where_cond(&nap, &pe)?;
            let e0 = lab.eq(0f32)?.where_cond(&self.point_embed[0].reshape((1, 1, 256))?.broadcast_as(shape)?, &zeros)?;
            let e1 = lab.eq(1f32)?.where_cond(&self.point_embed[1].reshape((1, 1, 256))?.broadcast_as(shape)?, &zeros)?;
            parts.push(((e + e0)? + e1)?);
        }
        if has_box {
            let mut c = Vec::with_capacity(b * 4);
            for p in ps {
                let bx = p.bbox.unwrap();
                c.extend([(bx[0] * sx + 0.5) / s, (bx[1] * sy + 0.5) / s, (bx[2] * sx + 0.5) / s, (bx[3] * sy + 0.5) / s]);
            }
            let c = Tensor::from_vec(c, (b, 2, 2), &self.dev)?;
            let pe = pe_encoding(&c, &self.gauss)?;
            let c0 = pe.i((.., 0..1, ..))?.broadcast_add(&self.point_embed[2].reshape((1, 1, 256))?)?;
            let c1 = pe.i((.., 1..2, ..))?.broadcast_add(&self.point_embed[3].reshape((1, 1, 256))?)?;
            parts.push(Tensor::cat(&[&c0, &c1], 1)?);
        }
        Ok(Tensor::cat(&parts, 1)?.contiguous()?)
    }
}

impl Backend for SamBackend {
    fn name(&self) -> String {
        match self.kind {
            ModelKind::VitB => "sam-vit-b".into(),
            ModelKind::Tiny => "mobile-sam".into(),
        }
    }

    fn encode(&self, rgb: &RgbImage) -> Result<Embedding> {
        let (w, h) = rgb.dimensions();
        if w == 0 || h == 0 {
            bail!("empty image");
        }
        let s = IMAGE_SIZE as f64 / w.max(h) as f64;
        let nw = ((w as f64 * s + 0.5).floor() as u32).clamp(1, IMAGE_SIZE as u32);
        let nh = ((h as f64 * s + 0.5).floor() as u32).clamp(1, IMAGE_SIZE as u32);
        let resized = if (nw, nh) == (w, h) { rgb.clone() } else { image::imageops::resize(rgb, nw, nh, image::imageops::FilterType::Triangle) };
        const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
        const STD: [f32; 3] = [58.395, 57.12, 57.375];
        let plane = IMAGE_SIZE * IMAGE_SIZE;
        let mut data = vec![0f32; 3 * plane];
        for (x, y, p) in resized.enumerate_pixels() {
            for c in 0..3 {
                data[c * plane + y as usize * IMAGE_SIZE + x as usize] = (p.0[c] as f32 - MEAN[c]) / STD[c];
            }
        }
        let t = Tensor::from_vec(data, (1, 3, IMAGE_SIZE, IMAGE_SIZE), &self.dev)?;
        let emb = match &self.encoder {
            Encoder::Vit(e) => e.forward(&t)?,
            Encoder::Tiny(e) => e.forward(&t)?,
        };
        Ok(Embedding { width: w, height: h, inner: Box::new(SamEmb { tensor: emb, input_w: nw as usize, input_h: nh as usize }) })
    }

    fn decode(&self, emb: &Embedding, prompts: &[Prompt]) -> Result<Vec<Proposal>> {
        let e = emb.inner.downcast_ref::<SamEmb>().expect("SAM embedding");
        let sx = e.input_w as f32 / emb.width as f32;
        let sy = e.input_h as f32 / emb.height as f32;
        // Prompts in one decoder call must share a shape: group by (point count, has box), keep order.
        let mut out: Vec<Option<Proposal>> = (0..prompts.len()).map(|_| None).collect();
        let mut groups: std::collections::BTreeMap<(usize, bool), Vec<usize>> = Default::default();
        for (i, p) in prompts.iter().enumerate() {
            if p.points.is_empty() && p.bbox.is_none() {
                out[i] = Some(Proposal { masks: [Mask::empty(), Mask::empty(), Mask::empty()], scores: [0.0; 3] });
                continue;
            }
            groups.entry((p.points.len(), p.bbox.is_some())).or_default().push(i);
        }
        // Several small decoder calls run side by side: many of candle's CPU element-wise ops are single-threaded,
        // so this uses the cores far better than one large batch.
        // One prompt per call, up to 16 calls at once (about 20 ms per prompt on a 16-core CPU); more threads
        // gain little and each call holds tens of MB.
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk = env_usize("MASKBOLT_DECODE_CHUNK", 1);
        let workers = env_usize("MASKBOLT_DECODE_THREADS", cores.min(16));
        let jobs: Vec<&[usize]> = groups.values().flat_map(|idx| idx.chunks(chunk)).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: std::sync::Mutex<Vec<(usize, Proposal)>> = std::sync::Mutex::new(Vec::with_capacity(prompts.len()));
        let first_err: std::sync::Mutex<Option<anyhow::Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|sc| {
            for _ in 0..workers.min(jobs.len()).max(1) {
                sc.spawn(|| loop {
                    let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if j >= jobs.len() || first_err.lock().unwrap().is_some() {
                        break;
                    }
                    match self.decode_job(e, emb, prompts, jobs[j], sx, sy) {
                        Ok(v) => results.lock().unwrap().extend(v),
                        Err(err) => {
                            first_err.lock().unwrap().get_or_insert(err);
                        }
                    }
                });
            }
        });
        if let Some(err) = first_err.into_inner().unwrap() {
            return Err(err);
        }
        for (i, p) in results.into_inner().unwrap() {
            out[i] = Some(p);
        }
        Ok(out.into_iter().map(|p| p.unwrap()).collect())
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(default)
}

fn pe_encoding(coords: &Tensor, gauss: &Tensor) -> Result<Tensor> {
    let c = coords.affine(2., -1.)?;
    let c = c.broadcast_matmul(gauss)?;
    let c = (c * (2. * std::f64::consts::PI))?;
    Ok(Tensor::cat(&[c.sin()?, c.cos()?], D::Minus1)?)
}

/// Bilinear upsampling (align_corners = false) of one 256x256 logit map to `w x h` pixels, threshold 0.
/// SAM's own post-processing goes 256 -> 1024, crops the padding, then resizes to the image; this maps each
/// pixel straight to the 256 grid, which differs from it by well under a pixel.
pub fn upsample(logits: &[f32], input_w: usize, input_h: usize, w: u32, h: u32) -> Mask {
    let cw = (input_w.div_ceil(4) + 1).min(LOW);
    let ch = (input_h.div_ceil(4) + 1).min(LOW);
    let (mut cx0, mut cy0, mut cx1, mut cy1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for cy in 0..ch {
        for cx in 0..cw {
            if logits[cy * LOW + cx] > 0.0 {
                cx0 = cx0.min(cx);
                cx1 = cx1.max(cx);
                cy0 = cy0.min(cy);
                cy1 = cy1.max(cy);
            }
        }
    }
    if cx0 == usize::MAX {
        return Mask::empty();
    }
    let sx = input_w as f32 / w as f32 / 4.0;
    let sy = input_h as f32 / h as f32 / 4.0;
    // Pixels whose sample point lies within one cell of the positive cells.
    let px = |c: f32, s: f32| (c + 0.5) / s - 0.5;
    let x_lo = px(cx0 as f32 - 1.0, sx).floor().max(0.0) as u32;
    let x_hi = (px(cx1 as f32 + 1.0, sx).ceil().max(0.0) as u32).min(w - 1);
    let y_lo = px(cy0 as f32 - 1.0, sy).floor().max(0.0) as u32;
    let y_hi = (px(cy1 as f32 + 1.0, sy).ceil().max(0.0) as u32).min(h - 1);
    if x_lo > x_hi || y_lo > y_hi {
        return Mask::empty();
    }
    let at = |cx: usize, cy: usize| logits[cy * LOW + cx];
    let xs: Vec<(usize, usize, f32)> = (x_lo..=x_hi)
        .map(|x| {
            let u = ((x as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (u.floor() as usize).min(LOW - 1);
            (x0, (x0 + 1).min(LOW - 1), u - x0 as f32)
        })
        .collect();
    Mask::from_fn(x_lo, y_lo, x_hi - x_lo + 1, y_hi - y_lo + 1, |x, y| {
        let v = ((y as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (v.floor() as usize).min(LOW - 1);
        let y1 = (y0 + 1).min(LOW - 1);
        let fy = v - y0 as f32;
        let (x0, x1, fx) = xs[(x - x_lo) as usize];
        let top = (1.0 - fx) * at(x0, y0) + fx * at(x1, y0);
        let bot = (1.0 - fx) * at(x0, y1) + fx * at(x1, y1);
        (1.0 - fy) * top + fy * bot > 0.0
    })
}
