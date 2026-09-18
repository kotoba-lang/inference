// ggml K-quant dot, workgroup-per-row form (M1 of root ADR-2609181800).
//
// Same arithmetic as ggml_kdot.wgsl (ggml's vec_dot_q4_K_q8_K / vec_dot_q6_K_q8_K
// reference: exact i32 accumulation inside each 256-element block, one f32
// multiply per block), but the WORK is laid out for a GPU instead of a CPU:
//
//   ggml_kdot.wgsl     : 1 thread = 1 output row, walks 256 values x blocks
//                        byte by byte. Kernel time on Intel Arc Pro B70 (wgpu
//                        -> Vulkan) 2026-09-18, [10240 x 2560] Q4_K: 0.80 ms =
//                        18 GB/s of weights (45 GB/s at 59 MB). The 17 ms that
//                        verify/metal_kdot.js prints is NOT this kernel: a
//                        single submit -> onSubmittedWorkDone round trip on
//                        Deno/wgpu costs 13-18 ms on that box at ANY size, so
//                        kdot_wg_parity.js times K dispatches in one command
//                        buffer and divides.
//   this file          : 1 workgroup (64 threads) = 1 output row. Thread t owns
//                        values t*4 .. t*4+3 of every block, which are four
//                        consecutive bytes of the same nibble half in Q4_K and
//                        four consecutive lanes of the same ql/qh words in
//                        Q6_K -- so each thread does ONE u32 load from the
//                        weight buffer and ONE from q8 per block instead of
//                        eight byte loads, and 64 threads stream a row
//                        together. Partials are reduced through workgroup
//                        memory. The float sum order across blocks and lanes
//                        differs from the reference, which is why the parity
//                        gate compares with a tolerance (2e-5) and not bitwise;
//                        the integer part inside a block is still exact.
//                        Kernel time on the same B70 and shape: 0.147 ms =
//                        100 GB/s (110 GB/s at 59 MB), 5.4x the reference; on an
//                        Apple M1 Max (Metal) 0.288 ms = 51 GB/s, 5.6x. Oracle:
//                        q4 -3.695018768310547 / q6 -3.704854965209961 against
//                        the ggml values -3.695021629333496 / -3.70485520362854.
//
// Bindings and Meta are identical to ggml_kdot.wgsl so the host can swap the
// module without touching bind groups. Dispatch: (rows, positions, 1)
// workgroups, NOT ceil(rows/64) -- one workgroup per row.

struct Meta {
  rows: u32,
  cols: u32,
  tensor_type: u32,
  positions: u32,
}

@group(0) @binding(0) var<uniform> params: Meta;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read> q8: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, 64>;

fn weight_byte(offset: u32) -> u32 {
  let word = weights[offset >> 2u];
  return (word >> ((offset & 3u) * 8u)) & 255u;
}

fn q8_byte(offset: u32) -> u32 {
  let word = q8[offset >> 2u];
  return (word >> ((offset & 3u) * 8u)) & 255u;
}

fn signed_i16_q8(offset: u32) -> i32 {
  let x = q8_byte(offset) | (q8_byte(offset + 1u) << 8u);
  return select(i32(x), i32(x) - 65536, x >= 32768u);
}

fn weight_half(offset: u32) -> f32 {
  let pair = unpack2x16float(weights[offset >> 2u]);
  return select(pair.x, pair.y, (offset & 2u) != 0u);
}

fn q8_float(offset: u32) -> f32 {
  return bitcast<f32>(q8[offset >> 2u]);
}

fn q4_scale(base: u32, j: u32) -> i32 {
  if j < 4u { return i32(weight_byte(base + 4u + j) & 63u); }
  return i32((weight_byte(base + 8u + j) & 15u) |
             ((weight_byte(base + j) >> 6u) << 4u));
}

fn q4_min(base: u32, j: u32) -> i32 {
  if j < 4u { return i32(weight_byte(base + 8u + j) & 63u); }
  return i32((weight_byte(base + 8u + j) >> 4u) |
             ((weight_byte(base + 4u + j) >> 6u) << 4u));
}

// Four signed q8 values packed in one word (byte offset must be 4-aligned).
fn q8_word_i8(offset: u32) -> vec4<i32> {
  let w = q8[offset >> 2u];
  return vec4<i32>(
    i32(w << 24u) >> 24u,
    i32(w << 16u) >> 24u,
    i32(w << 8u) >> 24u,
    i32(w) >> 24u);
}

// Four consecutive Q4_K values for `index .. index+3` of one block: they sit in
// four consecutive bytes of one 32-byte group, all low or all high nibbles.
fn q4_word_values(base: u32, index: u32) -> vec4<i32> {
  let group = index / 64u;
  let within = index & 63u;
  let w = weights[(base + 16u + group * 32u + (within & 31u)) >> 2u];
  let shift = select(0u, 4u, within >= 32u);
  return vec4<i32>(
    i32((w >> shift) & 15u),
    i32((w >> (8u + shift)) & 15u),
    i32((w >> (16u + shift)) & 15u),
    i32((w >> (24u + shift)) & 15u));
}

// A u32 at an arbitrary byte offset. Q4_K blocks are 144 bytes and Q8_K rows
// 292, both multiples of 4, so their words are aligned; Q6_K blocks are 210
// bytes, so from the second block on every word straddles two stored words.
fn weight_u32_at(offset: u32) -> u32 {
  let w0 = weights[offset >> 2u];
  let shift = (offset & 3u) * 8u;
  if shift == 0u { return w0; }
  let w1 = weights[(offset >> 2u) + 1u];
  return (w0 >> shift) | (w1 << (32u - shift));
}

// Four consecutive Q6_K values for `index .. index+3`: same ql word, same qh
// word, same nibble half and same qh bit pair.
fn q6_word_values(base: u32, index: u32) -> vec4<i32> {
  let group = index / 128u;
  let within = index & 127u;
  let lane = within & 31u;
  let ql_base = base + group * 64u;
  let qh = weight_u32_at(base + 128u + group * 32u + lane);
  let quarter = within / 32u;                  // 0..3
  let ql = weight_u32_at(ql_base + select(0u, 32u, (quarter & 1u) == 1u) + lane);
  let ql_shift = select(0u, 4u, quarter >= 2u); // quarters 2,3 read the high nibble
  let qh_shift = quarter * 2u;                  // bit pair per quarter
  var out: vec4<i32>;
  for (var b = 0u; b < 4u; b++) {
    let lo = (ql >> (b * 8u + ql_shift)) & 15u;
    let hi = ((qh >> (b * 8u + qh_shift)) & 3u) << 4u;
    out[b] = i32(lo | hi) - 32;
  }
  return out;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  let position = wg.y;
  let t = lid.x;
  let blocks = params.cols / 256u;
  let is_q4 = params.tensor_type == 12u;
  let block_bytes = select(210u, 144u, is_q4);
  let row_bytes = blocks * block_bytes;
  let index = t * 4u;                 // this thread's four values inside every block
  var acc = 0.0;                      // sum over blocks of d*yd*aux (thread's share)
  var mins = 0.0;                     // sum over blocks of dmin*yd*sumi (threads 0..15 only, Q4_K)

  if row < params.rows && position < params.positions {
    for (var block = 0u; block < blocks; block++) {
      let wb = row * row_bytes + block * block_bytes;
      let yb = (position * blocks + block) * 292u;
      let yd = q8_float(yb);
      let y = q8_word_i8(yb + 4u + index);
      if is_q4 {
        let j = index / 32u;
        let scale = q4_scale(wb, j);
        let q = q4_word_values(wb, index);
        let aux = scale * (y.x * q.x + y.y * q.y + y.z * q.z + y.w * q.w);
        acc += weight_half(wb) * yd * f32(aux);
        if t < 16u {
          // ggml: sumi = sum_j bsums[j] * mins[j/2]; sixteen terms, one per thread.
          mins += weight_half(wb + 2u) * yd * f32(signed_i16_q8(yb + 260u + t * 2u) * q4_min(wb, t / 2u));
        }
      } else {
        let j = index / 16u;
        let sb = weight_byte(wb + 192u + j);
        let scale = select(i32(sb), i32(sb) - 256, sb >= 128u);
        let q = q6_word_values(wb, index);
        let aux = scale * (y.x * q.x + y.y * q.y + y.z * q.z + y.w * q.w);
        acc += weight_half(wb + 208u) * yd * f32(aux);
      }
    }
  }
  partial[t] = acc - mins;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
    if t < stride { partial[t] += partial[t + stride]; }
    workgroupBarrier();
  }
  if t == 0u && row < params.rows && position < params.positions {
    output[position * params.rows + row] = partial[0];
  }
}
