//! SAM's mask decoder.
//!
//! Adapted from candle-transformers 0.11.0, src/models/segment_anything/mask_decoder.rs, part of candle
//! (https://github.com/huggingface/candle) by the candle authors, licensed MIT OR Apache-2.0; candle's
//! LICENSE-MIT and LICENSE-APACHE texts are reproduced in THIRD_PARTY.txt (sections 4 and 5). candle's code is
//! itself a port of Meta's segment-anything, Copyright (c) Meta Platforms, Inc. and affiliates, Apache-2.0.
//! This adapted file is distributed under the same terms as the original (MIT OR Apache-2.0) as part of Maskbolt.
//!
//! Changes from the original: the two 2x2 stride-2 transposed convolutions run as matrix products (candle's CPU
//! conv_transpose2d is a naive loop and dominated decoding time), GELU uses the exact erf form (as PyTorch's
//! nn.GELU), and only the three multimask outputs are computed.
use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::{linear, transformer::TwoWayTransformer, LayerNorm2d, Linear};

struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { hidden_dim };
            let out_dim = if i + 1 == num_layers { output_dim } else { hidden_dim };
            layers.push(linear(vb.pp(i), in_dim, out_dim, true)?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for (i, l) in self.layers.iter().enumerate() {
            xs = l.forward(&xs)?;
            if i + 1 < self.layers.len() {
                xs = xs.relu()?;
            }
        }
        Ok(xs)
    }
}

/// A ConvTranspose2d with kernel 2, stride 2, no padding, as one matmul.
struct Up2 {
    w: Tensor, // (cin, cout * 4), columns ordered (cout, dy, dx)
    b: Tensor, // (1, cout, 1, 1)
    cout: usize,
}

impl Up2 {
    fn new(cin: usize, cout: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((cin, cout, 2, 2), "weight")?.reshape((cin, cout * 4))?;
        let b = vb.get(cout, "bias")?.reshape((1, cout, 1, 1))?;
        Ok(Self { w, b, cout })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, cin, h, w) = xs.dims4()?;
        let x = xs.permute((0, 2, 3, 1))?.reshape((b * h * w, cin))?;
        let y = x.matmul(&self.w)?; // (b*h*w, cout*4)
        let y = y.reshape((b, h, w, self.cout, 2, 2))?.permute((0, 3, 1, 4, 2, 5))?.reshape((b, self.cout, 2 * h, 2 * w))?;
        y.broadcast_add(&self.b)
    }
}

pub struct MaskDecoder {
    iou_token: Tensor,
    mask_tokens: Tensor,
    iou_head: Mlp,
    up1: Up2,
    up_ln: LayerNorm2d,
    up2: Up2,
    hyper: Vec<Mlp>,
    transformer: TwoWayTransformer,
}

const NUM_MASK_TOKENS: usize = 4;

impl MaskDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let dim = 256;
        let iou_head = Mlp::new(dim, 256, NUM_MASK_TOKENS, 3, vb.pp("iou_prediction_head"))?;
        let iou_token = vb.pp("iou_token").get((1, dim), "weight")?;
        let mask_tokens = vb.pp("mask_tokens").get((NUM_MASK_TOKENS, dim), "weight")?;
        let up1 = Up2::new(dim, dim / 4, vb.pp("output_upscaling.0"))?;
        let up_ln = LayerNorm2d::new(dim / 4, 1e-6, vb.pp("output_upscaling.1"))?;
        let up2 = Up2::new(dim / 4, dim / 8, vb.pp("output_upscaling.3"))?;
        let vb_o = vb.pp("output_hypernetworks_mlps");
        let mut hyper = Vec::with_capacity(NUM_MASK_TOKENS);
        for i in 0..NUM_MASK_TOKENS {
            hyper.push(Mlp::new(dim, dim, dim / 8, 3, vb_o.pp(i))?);
        }
        let transformer = TwoWayTransformer::new(2, dim, 8, 2048, vb.pp("transformer"))?;
        Ok(Self { iou_token, mask_tokens, iou_head, up1, up_ln, up2, hyper, transformer })
    }

    /// Returns the three multimask logits (b, 3, 256, 256) and their IoU predictions (b, 3).
    pub fn forward(&self, image_embeddings: &Tensor, image_pe: &Tensor, sparse: &Tensor, dense: &Tensor) -> Result<(Tensor, Tensor)> {
        let b = sparse.dim(0)?;
        let out_tokens = Tensor::cat(&[&self.iou_token, &self.mask_tokens], 0)?;
        let (d1, d2) = out_tokens.dims2()?;
        let out_tokens = out_tokens.unsqueeze(0)?.expand((b, d1, d2))?;
        let tokens = Tensor::cat(&[&out_tokens, sparse], 1)?;
        let (_, c, h, w) = image_embeddings.dims4()?;
        let src = image_embeddings.broadcast_add(dense)?.expand((b, c, h, w))?.contiguous()?;
        let pos = image_pe.expand((b, c, h, w))?.contiguous()?;
        let (hs, src) = self.transformer.forward(&src, &pos, &tokens)?;
        let iou_out = hs.i((.., 0))?;
        let src = src.transpose(1, 2)?.reshape((b, c, h, w))?;
        let up = self.up1.forward(&src)?.apply(&self.up_ln)?.gelu_erf()?;
        let up = self.up2.forward(&up)?.gelu_erf()?;
        let mut hyper_in = Vec::with_capacity(3);
        for i in 1..NUM_MASK_TOKENS {
            hyper_in.push(self.hyper[i].forward(&hs.i((.., 1 + i))?)?);
        }
        let hyper_in = Tensor::stack(&hyper_in, 1)?.contiguous()?; // (b, 3, 32)
        let (b, uc, uh, uw) = up.dims4()?;
        let masks = hyper_in.matmul(&up.reshape((b, uc, uh * uw))?)?.reshape((b, 3, uh, uw))?;
        let iou = self.iou_head.forward(&iou_out)?.i((.., 1..))?;
        Ok((masks, iou))
    }
}
