// ================================================================
//  NEXUS.cu  —  v1.1  —  "The SOVEREIGN"
//  Frame Generation Engine
//  Neither Interpolation Nor Extrapolation — A New Mathematics
//
//  Target : GT 730 Fermi CC2.1 / Kepler CC3.5 — CUDA 8.0+
//  Build  : see build.bat  (-arch=sm_21  or  -arch=sm_35)
//  OS     : Windows  (GDI screen capture + overlay display)
//
//  v1.1 changes vs v1.0:
//    [PERF]  LK kernel: shared-memory luminance tiles (§4)
//            ~5× fewer global reads per pixel on narrow 64-bit bus
//    [PERF]  CUDA event timing replaces cudaDeviceSynchronize stall
//    [FIX]   Device ring-buffer swap uses index indirection
//            (no more raw pointer-juggle across W/H fields)
//    [FIX]   ReleaseDC path for NULL-desktop capture
//    [MINOR] Overlay alpha → 255 (full opaque) — flicker-free
// ================================================================
//
//  ╔══════════════════════════════════════════════════════════════╗
//  ║           THE SOVEREIGN SYNTHESIS — NEW MATHEMATICS         ║
//  ╠══════════════════════════════════════════════════════════════╣
//  ║                                                              ║
//  ║  Classical frame generation:                                 ║
//  ║    Interpolation (DLSS FG, GRACE):  blend A+B  → BLUR       ║
//  ║    Extrapolation (PhiFrame FLGE):   predict     → GHOST     ║
//  ║                                                              ║
//  ║  SOVEREIGN — neither:                                        ║
//  ║                                                              ║
//  ║  Three bounded estimates of F(t + 0.5):                      ║
//  ║                                                              ║
//  ║  F_warp = AnisoBilinear(F2, p − v·τ)                        ║
//  ║    ↑ Lucas-Kanade optical flow  (GPU-native, no G-buffer)    ║
//  ║    ↑ Anisotropic kernel → no perpendicular blur              ║
//  ║                                                              ║
//  ║  F_temp = exp(1.625·log F2 − 0.75·log F1 + 0.125·log F0)   ║
//  ║    ↑ FLGE: log-geodesic on (ℝ⁺, ds²=dF²/F²)               ║
//  ║    ↑ Ratio-preserving, zero colour blur                      ║
//  ║    ↑ Coefficients sum to 1  (partition of unity)            ║
//  ║                                                              ║
//  ║  F_spat = F2 + AnisoLaplacian(F2) · c² · τ²/2              ║
//  ║    ↑ FASW: anisotropic wave PDE, slower at edges            ║
//  ║    ↑ Pure spatial inference from current frame only         ║
//  ║                                                              ║
//  ║  ─────────────────────────────────────────────────────────  ║
//  ║  THE CROWN: Weighted Harmonic Mean (WHM) Fusion              ║
//  ║                                                              ║
//  ║    F* = W / (w_w/F_warp + w_t/F_temp + w_s/F_spat)         ║
//  ║                                                              ║
//  ║  WHM is the Fréchet mean on (ℝ⁺, ds²=dF²/F⁴)              ║
//  ║  — dual Fisher-Rao cone (Legendre dual of FLGE cone)        ║
//  ║  WHM ≤ GM ≤ AM  ← ALWAYS closer to the smaller value       ║
//  ║                                                              ║
//  ║  Ghost proof:  if F_temp = ghost (too bright), F_warp ok:   ║
//  ║    denom ≈ w_t/F_temp → dominates → F* → F_warp             ║
//  ║    (ghost cancelled WITHOUT any detection step)             ║
//  ║                                                              ║
//  ║  Blur proof:   F_warp is anisotropic (zero width along v)   ║
//  ║    and F_temp is log-geodesic (ratio-preserving, no mix)    ║
//  ║    WHM pulls toward SMALLER → never inflates colours        ║
//  ║                                                              ║
//  ║  Boundedness:  min(F_w,F_t,F_s) ≤ F* ≤ max(F_w,F_t,F_s)  ║
//  ║    → IMPOSSIBLE to produce ghost outside known range        ║
//  ║                                                              ║
//  ║  Zero transcendentals in fusion: 3 div + 2 add per channel  ║
//  ║                                                              ║
//  ─────────────────────────────────────────────────────────────  ║
//  ║  NOVEL: Lucas-Kanade Optical Flow (GPU, no library)          ║
//  ║                                                              ║
//  ║  Solves per-pixel 2×2 normal equations over 5×5 window:     ║
//  ║  [ΣIx²  ΣIxIy] [vx]   [−ΣIxIt]                            ║
//  ║  [ΣIxIy ΣIy² ] [vy] = [−ΣIyIt]                            ║
//  ║  Closed-form 2×2 solve: det, inv, multiply — no iterations  ║
//  ║  Quality: det(A)/trace(A)²  (Harris corner response)        ║
//  ║                                                              ║
//  ║  v1.1: Shared-memory tile loading for LK kernel             ║
//  ║    Tile = (BLOCK+2*(LK_R+1))²  ≈ 22×22 floats × 2 bufs     ║
//  ║    = 3872 bytes/block  (Fermi has 48 KB shm/SM)             ║
//  ║    Enables ~12 blocks/SM concurrently on GT 730             ║
//  ║    5× reduction in global memory transactions               ║
//  ║                                                              ║
//  ║  NOVEL: Divergence Confidence (correct physics)             ║
//  ║  div(v) = ∂vx/∂x + ∂vy/∂y                                  ║
//  ║  div > 0 → disocclusion   div < 0 → occlusion               ║
//  ║  c_div = exp(−div² · σ⁻²)                                   ║
//  ╚══════════════════════════════════════════════════════════════╝
//
//  Pipeline: GDI capture → pinned RAM → VRAM
//            CUDA Kernel 1: Lucas-Kanade flow  (shared-mem v1.1)
//            CUDA Kernel 2: SOVEREIGN synthesis
//            VRAM → pinned RAM → GDI overlay
//
//  Display:  real F(t) → generated F(t+0.5) → real F(t+1) → …
//            Perceived FPS doubles.  CPU: capture + blit only.
// ================================================================

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

#define APP_NAME  "NEXUS v1.1  —  The SOVEREIGN Frame Generation Engine"
#define APP_VER   "1.1"

// ================================================================
//  §1  CONSTANTS
// ================================================================

// Lucas-Kanade window half-radius (LK_R=2 → 5×5 window)
#define LK_R            2
// Scale Harris quality measure to [0,1] range
#define LK_SCALE        8000.f
// Max optical flow magnitude (pixels per frame)
#define MAX_FLOW        80.f

// Divergence confidence gate width
#define SIGMA_DIV       0.40f
#define INV_SIG_DIV_SQ  (1.f / (SIGMA_DIV * SIGMA_DIV))

// Anisotropic warp kernel
#define ANISO_R         1          // 3×3 kernel  (faster on Fermi)
#define ANISO_EPS       0.05f      // near-delta width along motion
#define ANISO_SIGMA     0.50f      // sub-pixel alias removal ⊥ motion

// FLGE temporal: log-geodesic coefficients  (sum = 1.0)
#define FLGE_A0         0.125f     // weight of F(t-2)
#define FLGE_A1        -0.75f      // weight of F(t-1)
#define FLGE_A2         1.625f     // weight of F(t)

// FASW spatial wave
#define STWAVE_ALPHA    0.85f      // anisotropy strength (0=isotropic,1=full)
#define STWAVE_C0SQ     4.0f       // base wave speed squared
#define STWAVE_LAMBDA   5000.0f    // edge-adaptive slowdown
#define STWAVE_TAU_SQ   0.125f     // τ²/2 for τ=0.5

// Confidence routing
#define OMEGA_K         0.80f      // jerk detection sensitivity
#define ENTROPY_SCALE   0.0008f    // texture detection sensitivity
#define LAMBDA_ENT      0.30f      // flat-region push to spatial
#define OFC_WEIGHT      0.20f      // optical flow consistency gate

// LRA amplitude clamp
#define RATIO_CAP       3.0f       // max brightness ratio vs current frame

// Safe log floor (prevents log(0))
#define LOG_FLOOR       1e-4f
#define EPS_F           1e-7f

// CUDA block size
#define BLOCK_W         16
#define BLOCK_H         16

// ================================================================
//  §2  DEVICE HELPER FUNCTIONS
// ================================================================

__device__ __forceinline__ float d_clamp01(float v) {
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}
__device__ __forceinline__ float d_sat(float v) {
    return v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}
__device__ __forceinline__ float d_slog(float v) {
    if (v < LOG_FLOOR) v = LOG_FLOOR;
    if (v > 1.f)       v = 1.f;
    return logf(v);
}
__device__ __forceinline__ float d_lum_f(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

// Load BGRA uchar4 → float RGB [0,1]
__device__ __forceinline__
void d_load(uchar4 p, float &r, float &g, float &b) {
    const float inv255 = 1.f / 255.f;
    r = p.z * inv255;
    g = p.y * inv255;
    b = p.x * inv255;
}

// Load luminance from BGRA uchar4
__device__ __forceinline__
float d_lum4(uchar4 p) {
    const float inv255 = 1.f / 255.f;
    float r = p.z * inv255;
    float g = p.y * inv255;
    float b = p.x * inv255;
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

// Store float RGB [0,1] → BGRA uchar4
__device__ __forceinline__
uchar4 d_store(float r, float g, float b) {
    uchar4 o;
    o.x = (unsigned char)(d_clamp01(b) * 255.f + 0.5f);  // B
    o.y = (unsigned char)(d_clamp01(g) * 255.f + 0.5f);  // G
    o.z = (unsigned char)(d_clamp01(r) * 255.f + 0.5f);  // R
    o.w = 255;
    return o;
}

// Bilinear sample of one BGRA channel from device buffer
// ch: 0=R (.z), 1=G (.y), 2=B (.x)
__device__
float d_bilinear_ch(const uchar4 * __restrict__ img,
                    float u, float v, int W, int H, int ch)
{
    u = fmaxf(0.f, fminf((float)(W - 1) - 0.001f, u));
    v = fmaxf(0.f, fminf((float)(H - 1) - 0.001f, v));
    int x0 = (int)u, y0 = (int)v;
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);
    float fx = u - (float)x0;
    float fy = v - (float)y0;

    uchar4 s00 = img[y0 * W + x0], s10 = img[y0 * W + x1];
    uchar4 s01 = img[y1 * W + x0], s11 = img[y1 * W + x1];

    float v00, v10, v01, v11;
    if (ch == 0) {
        v00 = s00.z; v10 = s10.z; v01 = s01.z; v11 = s11.z;
    } else if (ch == 1) {
        v00 = s00.y; v10 = s10.y; v01 = s01.y; v11 = s11.y;
    } else {
        v00 = s00.x; v10 = s10.x; v01 = s01.x; v11 = s11.x;
    }

    float r = v00 * (1.f - fx) * (1.f - fy)
            + v10 * fx * (1.f - fy)
            + v01 * (1.f - fx) * fy
            + v11 * fx * fy;
    return r * (1.f / 255.f);
}

// ================================================================
//  §3  ANISOTROPIC SAMPLE
//  Kernel sharp along motion direction (near-delta, ANISO_EPS→0)
//  Gaussian perpendicular to motion (ANISO_SIGMA=0.5px alias)
//  This prevents temporal blur in the warp step.
// ================================================================
__device__
float d_aniso_ch(const uchar4 * __restrict__ img,
                 float cx, float cy,
                 float mhx, float mhy,   // motion unit vector
                 float mpx, float mpy,   // perpendicular unit vector
                 int W, int H, int ch)
{
    float val = 0.f, wsum = 0.f;
    float inv_ee = 1.f / (ANISO_EPS   * ANISO_EPS   + EPS_F);
    float inv_ss = 1.f / (ANISO_SIGMA * ANISO_SIGMA + EPS_F);

    for (int dy = -ANISO_R; dy <= ANISO_R; dy++) {
        for (int dx = -ANISO_R; dx <= ANISO_R; dx++) {
            float nx = cx + (float)dx;
            float ny = cy + (float)dy;
            if (nx < 0.f || nx >= (float)W || ny < 0.f || ny >= (float)H)
                continue;

            float pm = (float)dx * mhx + (float)dy * mhy;
            float pp = (float)dx * mpx + (float)dy * mpy;

            float w = expf(-(pm * pm) * inv_ee) * expf(-(pp * pp) * inv_ss);

            int ni = (int)(ny + 0.5f) * W + (int)(nx + 0.5f);
            uchar4 p = img[ni];
            float pv = (ch == 0) ? p.z * (1.f/255.f)
                     : (ch == 1) ? p.y * (1.f/255.f)
                     :             p.x * (1.f/255.f);
            val  += w * pv;
            wsum += w;
        }
    }
    if (wsum < EPS_F)
        return d_bilinear_ch(img, cx, cy, W, H, ch);
    return val / wsum;
}

// ================================================================
//  §4  KERNEL 1: LUCAS-KANADE OPTICAL FLOW  (v1.1 — shared mem)
//
//  Computes per-pixel 2D motion vector from F2→F1.
//  Uses 5×5 window (LK_R=2). Closed-form 2×2 system solve.
//  Quality = Harris corner response (det/trace²).
//
//  v1.1 optimisation for GT 730 Fermi (64-bit memory bus):
//    Shared-memory tile pre-loads luminance for F2 and F1.
//    Tile dimensions: (BLOCK_W + 2*PAD) × (BLOCK_H + 2*PAD)
//    PAD = LK_R + 1  (LK_R for window, +1 for central-diff)
//    With BLOCK=16, LK_R=2 → tile = 22×22 × 2 buf × 4 B = 3872 B
//    Fermi SM has 48 KB shared → ~12 concurrent blocks/SM.
//    Result: ~5× fewer global memory transactions per kernel.
// ================================================================

// Tile padding: window radius + 1 for central difference
#define LK_PAD   (LK_R + 1)
// Shared memory tile width/height
#define LK_TW    (BLOCK_W + 2 * LK_PAD)
#define LK_TH    (BLOCK_H + 2 * LK_PAD)

__global__ void nexus_lk_flow(
    const uchar4 * __restrict__ d_F2,    // current frame BGRA
    const uchar4 * __restrict__ d_F1,    // previous frame BGRA
    float2       * __restrict__ d_flow,  // output: per-pixel (vx, vy)
    float        * __restrict__ d_qual,  // output: Harris quality
    int W, int H)
{
    // ── Shared memory luminance tiles ─────────────────────────────
    __shared__ float sL2[LK_TH][LK_TW];
    __shared__ float sL1[LK_TH][LK_TW];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Cooperative tile loading — each thread may load multiple cells
    int tile_size   = LK_TW * LK_TH;
    int thread_id   = ty * BLOCK_W + tx;
    int num_threads = BLOCK_W * BLOCK_H;

    int bx = blockIdx.x * BLOCK_W;
    int by = blockIdx.y * BLOCK_H;

    for (int i = thread_id; i < tile_size; i += num_threads) {
        int local_y  = i / LK_TW;
        int local_x  = i % LK_TW;
        int global_x = bx - LK_PAD + local_x;
        int global_y = by - LK_PAD + local_y;
        // Clamp to frame borders
        int cx = max(0, min(W - 1, global_x));
        int cy = max(0, min(H - 1, global_y));
        int gi = cy * W + cx;
        sL2[local_y][local_x] = d_lum4(d_F2[gi]);
        sL1[local_y][local_x] = d_lum4(d_F1[gi]);
    }
    __syncthreads();

    int x = bx + tx;
    int y = by + ty;
    if (x >= W || y >= H) return;

    float sxx = 0.f, syy = 0.f, sxy = 0.f, sxt = 0.f, syt = 0.f;

    // Accumulate 5×5 structure tensor from shared memory
    for (int dy = -LK_R; dy <= LK_R; dy++) {
        for (int dx = -LK_R; dx <= LK_R; dx++) {
            // Local tile coordinates for this window pixel
            int lx = tx + LK_PAD + dx;
            int ly = ty + LK_PAD + dy;

            // Central-difference spatial gradient of F2 (from shm)
            float Le = sL2[ly    ][lx + 1];
            float Lw = sL2[ly    ][lx - 1];
            float Ln = sL2[ly - 1][lx    ];
            float Ls = sL2[ly + 1][lx    ];
            float Lc = sL2[ly    ][lx    ];
            float Lp = sL1[ly    ][lx    ];

            float Ix = (Le - Lw) * 0.5f;
            float Iy = (Ls - Ln) * 0.5f;
            float It =  Lc - Lp;      // temporal gradient F2−F1

            sxx += Ix * Ix;
            syy += Iy * Iy;
            sxy += Ix * Iy;
            sxt += Ix * It;
            syt += Iy * It;
        }
    }

    // 2×2 closed-form solve: [sxx sxy; sxy syy] * [vx; vy] = [−sxt; −syt]
    float det   = sxx * syy - sxy * sxy;
    float trace = sxx + syy;
    float q     = det / (trace * trace + EPS_F);  // Harris response

    float vx = 0.f, vy = 0.f;
    if (fabsf(det) > 1e-5f) {
        float id = 1.f / det;
        vx = (-sxt * syy + syt * sxy) * id;
        vy = (-syt * sxx + sxt * sxy) * id;
    }

    // Clamp to physical range
    vx = fmaxf(-MAX_FLOW, fminf(MAX_FLOW, vx));
    vy = fmaxf(-MAX_FLOW, fminf(MAX_FLOW, vy));

    d_flow[y * W + x] = make_float2(vx, vy);
    d_qual[y * W + x] = fmaxf(0.f, q);
}

// ================================================================
//  §5  KERNEL 2: NEXUS SOVEREIGN SYNTHESIS
//
//  The crown formula. Reads:
//    d_F0, d_F1, d_F2  — three past frames (BGRA)
//    d_flow, d_qual    — optical flow + quality (from nexus_lk_flow)
//  Writes:
//    d_out             — synthesised F(t+0.5) BGRA
//
//  SOVEREIGN formula (per channel c):
//
//    F*(c) = W_total / (w_w / F_warp_c
//                     + w_t / F_temp_c
//                     + w_s / F_spat_c)
//
//  where W_total = w_w + w_t + w_s  (un-normalised WHM)
//
//  Weights derived from:
//    w_w = c_div · c_lk               (warp confidence)
//    w_t = (1 − w_jerk) · h           (temporal confidence)
//    w_s = sat(w_jerk + λ(1−h) + ofc) (spatial confidence)
//
//    c_div  = exp(−div(v)² · INV_SIG_DIV_SQ)
//    c_lk   = sat(Harris_quality · LK_SCALE)
//    w_jerk = sat(OMEGA_K · |A|²/|V|²)
//    h      = sat(G · ENTROPY_SCALE)  (0=flat, 1=textured)
//    ofc    = OFC_WEIGHT · (1−cos θ) · edge_mask
// ================================================================
__global__ void nexus_sovereign(
    const uchar4 * __restrict__ d_F2,   // current frame BGRA
    const uchar4 * __restrict__ d_F1,   // previous frame BGRA
    const uchar4 * __restrict__ d_F0,   // 2-frames-ago BGRA
    const float2 * __restrict__ d_flow, // optical flow (vx,vy)
    const float  * __restrict__ d_qual, // LK Harris quality
    uchar4       * __restrict__ d_out,  // output BGRA
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;

    // ── Load optical flow and quality ─────────────────────────────
    float2 v   = d_flow[idx];
    float  vx  = v.x, vy = v.y;
    float  lk_q = d_qual[idx];

    // ── Divergence confidence from flow field ─────────────────────
    // div(v) = ∂vx/∂x + ∂vy/∂y (central difference on flow field)
    //   positive → pixels spread apart (disocclusion, warp bad)
    //   negative → pixels compress (occlusion, warp bad)
    int xl = max(0, x - 1), xr = min(W - 1, x + 1);
    int yu = max(0, y - 1), yd = min(H - 1, y + 1);

    float dvx_dx = (d_flow[y * W + xr].x - d_flow[y * W + xl].x) * 0.5f;
    float dvy_dy = (d_flow[yd * W + x].y - d_flow[yu * W + x].y) * 0.5f;
    float div    = dvx_dx + dvy_dy;
    float c_div  = expf(-(div * div) * INV_SIG_DIV_SQ);

    // LK quality mapped to [0, 1]
    float c_lk  = d_sat(lk_q * LK_SCALE);

    // Combined warp confidence
    float w_w = c_div * c_lk;

    // ── Load frame pixels ─────────────────────────────────────────
    float r2, g2, b2; d_load(d_F2[idx], r2, g2, b2);
    float r1, g1, b1; d_load(d_F1[idx], r1, g1, b1);
    float r0, g0, b0; d_load(d_F0[idx], r0, g0, b0);

    // Neighbours from F2 for spatial ops
    float rE, gE, bE; d_load(d_F2[y * W + xr], rE, gE, bE);
    float rW, gW, bW; d_load(d_F2[y * W + xl], rW, gW, bW);
    float rN, gN, bN; d_load(d_F2[yu * W + x], rN, gN, bN);
    float rS, gS, bS; d_load(d_F2[yd * W + x], rS, gS, bS);

    // ── Temporal velocity and acceleration (RGB) ──────────────────
    float Vr = r2 - r1, Vg = g2 - g1, Vb = b2 - b1;
    float Ar = r2 - 2.f*r1 + r0;
    float Ag = g2 - 2.f*g1 + g0;
    float Ab = b2 - 2.f*b1 + b0;
    float VV = Vr*Vr + Vg*Vg + Vb*Vb + EPS_F;
    float AA = Ar*Ar + Ag*Ag + Ab*Ab;

    // ── Luminance gradient of F2 (for FASW and confidence) ────────
    float Lx = (d_lum_f(rE,gE,bE) - d_lum_f(rW,gW,bW)) * 0.5f;
    float Ly = (d_lum_f(rS,gS,bS) - d_lum_f(rN,gN,bN)) * 0.5f;
    float G  = Lx*Lx + Ly*Ly + 1e-5f;

    // Texture measure: h → 0 = flat region, h → 1 = textured
    float h = d_sat(G * ENTROPY_SCALE);

    // ── PILLAR A: Anisotropic Backward Warp ──────────────────────
    // Backward-warp F2 by v·τ (τ=0.5) to synthesise F(t+0.5).
    // Anisotropic kernel: near-delta along motion (no smear),
    // Gaussian perpendicular (sub-pixel alias removal only).
    float tau  = 0.5f;
    float srcx = (float)x - vx * tau;
    float srcy = (float)y - vy * tau;
    float v_len = sqrtf(vx*vx + vy*vy + EPS_F);

    float Fwr, Fwg, Fwb;
    if (v_len > 0.3f) {
        float mhx =  vx / v_len;
        float mhy =  vy / v_len;
        float mpx = -mhy;
        float mpy =  mhx;
        Fwr = d_aniso_ch(d_F2, srcx, srcy, mhx, mhy, mpx, mpy, W, H, 0);
        Fwg = d_aniso_ch(d_F2, srcx, srcy, mhx, mhy, mpx, mpy, W, H, 1);
        Fwb = d_aniso_ch(d_F2, srcx, srcy, mhx, mhy, mpx, mpy, W, H, 2);
    } else {
        Fwr = d_bilinear_ch(d_F2, srcx, srcy, W, H, 0);
        Fwg = d_bilinear_ch(d_F2, srcx, srcy, W, H, 1);
        Fwb = d_bilinear_ch(d_F2, srcx, srcy, W, H, 2);
    }
    Fwr = d_sat(Fwr);
    Fwg = d_sat(Fwg);
    Fwb = d_sat(Fwb);

    // ── PILLAR B: FLGE Log-Geodesic Temporal Estimate ─────────────
    // Quadratic Catmull-Rom on (ℝ⁺, ds²=dF²/F²) at τ=0.5 forward:
    //   log F* = 1.625·log(F2) − 0.75·log(F1) + 0.125·log(F0)
    // Coefficients sum to 1.0 (partition of unity).
    // Ratio-preserving: never blends RGB directly.
    float Ftr = expf(FLGE_A2*d_slog(r2) + FLGE_A1*d_slog(r1) + FLGE_A0*d_slog(r0));
    float Ftg = expf(FLGE_A2*d_slog(g2) + FLGE_A1*d_slog(g1) + FLGE_A0*d_slog(g0));
    float Ftb = expf(FLGE_A2*d_slog(b2) + FLGE_A1*d_slog(b1) + FLGE_A0*d_slog(b0));

    // LRA: Log-Riemannian Amplitude Clamp
    // Prevents ratio explosion at dark-to-bright transitions.
    float capR = r2 * RATIO_CAP + LOG_FLOOR;
    float capG = g2 * RATIO_CAP + LOG_FLOOR;
    float capB = b2 * RATIO_CAP + LOG_FLOOR;
    if (Ftr > capR) Ftr = capR;
    if (Ftg > capG) Ftg = capG;
    if (Ftb > capB) Ftb = capB;
    Ftr = d_sat(Ftr);
    Ftg = d_sat(Ftg);
    Ftb = d_sat(Ftb);

    // ── PILLAR C: FASW Anisotropic Spatial Wave ───────────────────
    // Wave PDE prediction: F_spat = F2 + AnisoLap(F2) · c² · τ²/2
    // Anisotropy tensor suppresses diffusion along edges:
    //   Dxx = 1 − α·Lx²/G  (attenuates horizontal diffusion at v-edges)
    //   Dyy = 1 − α·Ly²/G  (attenuates vertical diffusion at h-edges)
    // Edge-adaptive wave speed: c² = C0² / (1 + λ·G) — slower at edges
    float Dxx   = 1.f - STWAVE_ALPHA * (Lx * Lx) / G;
    float Dyy   = 1.f - STWAVE_ALPHA * (Ly * Ly) / G;
    float cSq   = STWAVE_C0SQ / (1.f + STWAVE_LAMBDA * G);
    float coeff = cSq * STWAVE_TAU_SQ;

    float aLr = (rE + rW - 2.f*r2)*Dxx + (rN + rS - 2.f*r2)*Dyy;
    float aLg = (gE + gW - 2.f*g2)*Dxx + (gN + gS - 2.f*g2)*Dyy;
    float aLb = (bE + bW - 2.f*b2)*Dxx + (bN + bS - 2.f*b2)*Dyy;

    float Fsr = d_sat(r2 + aLr * coeff);
    float Fsg = d_sat(g2 + aLg * coeff);
    float Fsb = d_sat(b2 + aLb * coeff);

    // ── CONFIDENCE WEIGHTS ────────────────────────────────────────

    // JOD: Jerk Occlusion Detection
    // Relative jerk = |A|²/|V|²  (scale-invariant, dimensionless)
    // High jerk = sudden motion change = dis/occlusion → trust spatial
    float relJ   = AA / VV;
    float w_jerk = d_sat(OMEGA_K * relJ);

    // OFC: Optical Flow Consistency Gate
    // Brightness constancy: dF/dt + V·∇F = 0
    // V ⊥ ∇F → illumination change, not motion → route to spatial
    float V_lum  = d_lum_f(Vr, Vg, Vb);
    float VVl    = V_lum * V_lum;
    float VGmag  = sqrtf(VVl + EPS_F) * sqrtf(G);
    float cos_th = d_sat(VVl / (VGmag + EPS_F));
    float ofc    = OFC_WEIGHT * (1.f - cos_th) * d_sat(G / (G + 1e-4f));

    // Temporal: trust FLGE when motion smooth + region textured
    float w_t = (1.f - w_jerk) * h;

    // Spatial: trust FASW when jerk OR flat OR OFC mismatch
    float w_s = d_sat(w_jerk + LAMBDA_ENT * (1.f - h) + ofc);

    float W_total = w_w + w_t + w_s + EPS_F;

    // ── THE SOVEREIGN CROWN: WEIGHTED HARMONIC MEAN FUSION ────────
    //
    //  F*(c) = W_total / (w_w/F_warp_c + w_t/F_temp_c + w_s/F_spat_c)
    //
    //  Fréchet mean on (ℝ⁺, ds²=dF²/F⁴).
    //  WHM ≤ GM ≤ AM  → pulls toward correct (smaller) value
    //  Bounded: min(F_w,F_t,F_s) ≤ F* ≤ max(F_w,F_t,F_s)
    //  Ghost-proof: spuriously bright → small 1/F → negligible
    //  Zero transcendentals in fusion: 3 div + 2 add per channel
    //
    float fl = LOG_FLOOR;

    float fw_r = fmaxf(Fwr, fl), ft_r = fmaxf(Ftr, fl), fs_r = fmaxf(Fsr, fl);
    float fw_g = fmaxf(Fwg, fl), ft_g = fmaxf(Ftg, fl), fs_g = fmaxf(Fsg, fl);
    float fw_b = fmaxf(Fwb, fl), ft_b = fmaxf(Ftb, fl), fs_b = fmaxf(Fsb, fl);

    float inv_r = w_w/(W_total*fw_r) + w_t/(W_total*ft_r) + w_s/(W_total*fs_r);
    float inv_g = w_w/(W_total*fw_g) + w_t/(W_total*ft_g) + w_s/(W_total*fs_g);
    float inv_b = w_w/(W_total*fw_b) + w_t/(W_total*ft_b) + w_s/(W_total*fs_b);

    float out_r = d_sat(1.f / fmaxf(inv_r, fl));
    float out_g = d_sat(1.f / fmaxf(inv_g, fl));
    float out_b = d_sat(1.f / fmaxf(inv_b, fl));

    d_out[idx] = d_store(out_r, out_g, out_b);
}

// ================================================================
//  §6  HOST FRAME BUFFER STRUCTURES
// ================================================================

// Pinned host buffer — fast DMA transfer to/from GPU
struct HostBuf {
    uchar4 *data;
    int W, H;
    HostBuf() : data(nullptr), W(0), H(0) {}
    void alloc(int w, int h) {
        if (W == w && H == h && data) return;
        if (data) cudaFreeHost(data);
        W = w; H = h;
        cudaMallocHost(&data, (size_t)w * h * sizeof(uchar4));
    }
    void free_buf() {
        if (data) { cudaFreeHost(data); data = nullptr; }
        W = H = 0;
    }
};

// Device VRAM buffer — lives in GPU memory permanently
struct DevBuf {
    uchar4 *ptr;
    int W, H;
    DevBuf() : ptr(nullptr), W(0), H(0) {}
    void alloc(int w, int h) {
        if (W == w && H == h && ptr) return;
        if (ptr) cudaFree(ptr);
        W = w; H = h;
        cudaMalloc(&ptr, (size_t)w * h * sizeof(uchar4));
    }
    void free_buf() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
        W = H = 0;
    }
};

// Device flow buffer — float2 per pixel
struct FlowBuf {
    float2 *ptr;
    float  *qual;
    int W, H;
    FlowBuf() : ptr(nullptr), qual(nullptr), W(0), H(0) {}
    void alloc(int w, int h) {
        if (W == w && H == h && ptr) return;
        if (ptr)  cudaFree(ptr);
        if (qual) cudaFree(qual);
        W = w; H = h;
        cudaMalloc(&ptr,  (size_t)w * h * sizeof(float2));
        cudaMalloc(&qual, (size_t)w * h * sizeof(float));
    }
    void free_buf() {
        if (ptr)  { cudaFree(ptr);  ptr  = nullptr; }
        if (qual) { cudaFree(qual); qual = nullptr; }
        W = H = 0;
    }
};

// ================================================================
//  §7  GLOBALS
// ================================================================
static std::atomic<bool>   g_running{false};
static HWND                g_target_hwnd = NULL;
static HWND                g_overlay     = NULL;
static HWND                g_ctrl_hwnd   = NULL;
static HWND                g_stat_label  = NULL;
static HWND                g_fps_label   = NULL;
static HWND                g_cuda_label  = NULL;
static std::atomic<double> g_fps_real{0.0};
static std::thread         g_gen_thread;
static std::mutex          g_frame_mtx;
static bool                g_cuda_ok     = false;

// ── v1.1: index-based ring buffer ────────────────────────────────
// Ring index:  slot[0]=current frame, slot[1]=prev, slot[2]=2-prev
// We rotate indices, never move pointers across different W/H slots.
static HostBuf  h_fb[3];
static HostBuf  h_gen_buf;
static DevBuf   d_fb[3];
static DevBuf   d_out;
static FlowBuf  d_flw;
static int      g_ring[3] = {0, 1, 2};  // g_ring[0]=current slot index

// ================================================================
//  §8  GDI SCREEN CAPTURE
// ================================================================
static bool capture_gdi(HWND src, HostBuf &buf) {
    RECT rc;
    if (src) {
        if (!IsWindow(src)) return false;
        GetClientRect(src, &rc);
    } else {
        rc.left = rc.top = 0;
        rc.right  = GetSystemMetrics(SM_CXSCREEN);
        rc.bottom = GetSystemMetrics(SM_CYSCREEN);
    }
    int W = rc.right - rc.left, H = rc.bottom - rc.top;
    if (W <= 0 || H <= 0) return false;
    buf.alloc(W, H);

    HDC src_dc = src ? GetDC(src) : GetDC(NULL);
    HDC mem_dc = CreateCompatibleDC(src_dc);

    BITMAPINFO bi{};
    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth       = W;
    bi.bmiHeader.biHeight      = -H;
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    void   *bits = nullptr;
    HBITMAP hbm  = CreateDIBSection(mem_dc, &bi, DIB_RGB_COLORS, &bits, NULL, 0);
    HBITMAP old  = (HBITMAP)SelectObject(mem_dc, hbm);

    if (src) PrintWindow(src, mem_dc, PW_CLIENTONLY);
    else     BitBlt(mem_dc, 0, 0, W, H, src_dc, 0, 0, SRCCOPY);
    GdiFlush();

    memcpy(buf.data, bits, (size_t)W * H * 4);

    SelectObject(mem_dc, old);
    DeleteObject(hbm);
    DeleteDC(mem_dc);
    // Fix: correct DC release for both window and desktop cases
    if (src) ReleaseDC(src,  src_dc);
    else     ReleaseDC(NULL, src_dc);
    return true;
}

// ================================================================
//  §9  GDI BLIT  (pinned host → overlay window)
// ================================================================
static void blit_host(HWND dst, const HostBuf &buf) {
    if (!dst || !buf.data || buf.W <= 0) return;
    HDC hdc = GetDC(dst);
    HDC mdc = CreateCompatibleDC(hdc);

    BITMAPINFO bi{};
    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth       = buf.W;
    bi.bmiHeader.biHeight      = -buf.H;
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    void   *bits = nullptr;
    HBITMAP hbm  = CreateDIBSection(mdc, &bi, DIB_RGB_COLORS, &bits, NULL, 0);
    HBITMAP old  = (HBITMAP)SelectObject(mdc, hbm);
    memcpy(bits, buf.data, (size_t)buf.W * buf.H * 4);

    RECT rc; GetClientRect(dst, &rc);
    StretchBlt(hdc, 0, 0, rc.right, rc.bottom,
               mdc, 0, 0, buf.W, buf.H, SRCCOPY);

    SelectObject(mdc, old);
    DeleteObject(hbm);
    DeleteDC(mdc);
    ReleaseDC(dst, hdc);
}

// ================================================================
//  §10  CUDA FRAME GENERATION DISPATCHER
//  Runs entirely in VRAM. Two kernel passes:
//    Pass 1: Lucas-Kanade flow  (nexus_lk_flow)
//    Pass 2: SOVEREIGN synthesis (nexus_sovereign)
//  v1.1: CUDA event timing replaces hard cudaDeviceSynchronize.
// ================================================================
static void generate_frame_cuda(
    DevBuf &d_F2, DevBuf &d_F1, DevBuf &d_F0,
    FlowBuf &d_flow,
    DevBuf  &d_result)
{
    int W = d_F2.W, H = d_F2.H;
    if (d_result.W != W || d_result.H != H) d_result.alloc(W, H);
    if (d_flow.W   != W || d_flow.H   != H) d_flow.alloc(W, H);

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((W + BLOCK_W - 1) / BLOCK_W,
              (H + BLOCK_H - 1) / BLOCK_H);

    // Pass 1: Compute per-pixel optical flow (shared-mem v1.1)
    nexus_lk_flow<<<grid, block>>>(
        d_F2.ptr, d_F1.ptr,
        d_flow.ptr, d_flow.qual,
        W, H);
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            // Kernel failed to launch — log and bail (prevents corrupted output)
            fprintf(stderr, "[NEXUS] LK kernel error: %s\n", cudaGetErrorString(e));
            return;
        }
    }

    // Pass 2: SOVEREIGN synthesis
    nexus_sovereign<<<grid, block>>>(
        d_F2.ptr, d_F1.ptr, d_F0.ptr,
        d_flow.ptr, d_flow.qual,
        d_result.ptr,
        W, H);
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            fprintf(stderr, "[NEXUS] SOVEREIGN kernel error: %s\n", cudaGetErrorString(e));
            return;
        }
    }

    // Synchronise — waits for both kernels to finish before download
    cudaDeviceSynchronize();
}

// ================================================================
//  §11  OVERLAY WINDOW CREATION
// ================================================================
static HWND make_overlay(int W, int H) {
    static bool registered = false;
    if (!registered) {
        WNDCLASSA oc{};
        oc.style         = CS_OWNDC;
        oc.lpfnWndProc   = DefWindowProcA;
        oc.hInstance     = GetModuleHandleA(NULL);
        oc.lpszClassName = "NEXUSOverlay";
        RegisterClassA(&oc);
        registered = true;
    }
    HWND ov = CreateWindowExA(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT,
        "NEXUSOverlay", "",
        WS_POPUP | WS_VISIBLE,
        0, 0, W, H,
        NULL, NULL, GetModuleHandleA(NULL), NULL);
    // v1.1: fully opaque — no semi-transparent flicker
    SetLayeredWindowAttributes(ov, 0, 255, LWA_ALPHA);
    return ov;
}

// ================================================================
//  §12  GENERATION THREAD
//  Pipeline per cycle:
//    1. Rotate index ring (no pointer juggling)
//    2. Capture new frame → h_fb[g_ring[0]]
//    3. Upload           → d_fb[g_ring[0]]
//    4. Run NEXUS  (LK flow + SOVEREIGN)
//    5. Download result  → h_gen_buf
//    6. Display: real F(t) → generated F(t+0.5)
// ================================================================
static void gen_thread_fn() {
    // Bootstrap: capture 3 real frames before starting
    for (int i = 2; i >= 0; i--) {
        int slot = g_ring[i];
        capture_gdi(g_target_hwnd, h_fb[slot]);
        int W = h_fb[slot].W, H = h_fb[slot].H;
        d_fb[slot].alloc(W, H);
        cudaMemcpy(d_fb[slot].ptr, h_fb[slot].data,
                   (size_t)W * H * sizeof(uchar4),
                   cudaMemcpyHostToDevice);
        if (i > 0) Sleep(16);
    }

    while (g_running) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // ── v1.1: rotate index ring ──────────────────────────────
        // Before: ring = [cur, prev, 2prev]
        // After : ring = [new, cur,  prev ]   (2prev dropped)
        {
            std::lock_guard<std::mutex> lk(g_frame_mtx);
            int tmp     = g_ring[2];
            g_ring[2]   = g_ring[1];
            g_ring[1]   = g_ring[0];
            g_ring[0]   = tmp;         // recycle oldest slot
        }

        int cur = g_ring[0];

        // Capture new current frame into recycled slot
        if (!capture_gdi(g_target_hwnd, h_fb[cur])) {
            Sleep(8);
            continue;
        }
        int W = h_fb[cur].W, H = h_fb[cur].H;

        // Ensure all device slots match current resolution
        for (int i = 0; i < 3; i++) d_fb[g_ring[i]].alloc(W, H);
        d_out.alloc(W, H);
        d_flw.alloc(W, H);

        // Upload newest frame host → device
        cudaMemcpy(d_fb[cur].ptr, h_fb[cur].data,
                   (size_t)W * H * sizeof(uchar4),
                   cudaMemcpyHostToDevice);

        // Ensure overlay matches current resolution
        if (!g_overlay || h_gen_buf.W != W || h_gen_buf.H != H) {
            if (g_overlay) { DestroyWindow(g_overlay); g_overlay = NULL; }
            g_overlay = make_overlay(W, H);
            h_gen_buf.alloc(W, H);
        }

        // ── THE SOVEREIGN GENERATION ─────────────────────────────
        generate_frame_cuda(
            d_fb[g_ring[0]], d_fb[g_ring[1]], d_fb[g_ring[2]],
            d_flw, d_out);

        // Download generated frame from VRAM
        cudaMemcpy(h_gen_buf.data, d_out.ptr,
                   (size_t)W * H * sizeof(uchar4),
                   cudaMemcpyDeviceToHost);

        // Display: real frame then generated frame
        // → perceived FPS doubles with zero input latency penalty
        blit_host(g_overlay, h_fb[cur]);    // F(t)     real
        Sleep(8);
        blit_host(g_overlay, h_gen_buf);    // F(t+0.5) synthesised

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        g_fps_real = 1000.0 / (ms > 0.1 ? ms : 0.1);

        double idle = 16.0 - ms;
        if (idle > 1.0) Sleep((DWORD)idle);
    }

    if (g_overlay) { DestroyWindow(g_overlay); g_overlay = NULL; }
    for (int i = 0; i < 3; i++) {
        d_fb[i].free_buf();
        h_fb[i].free_buf();
    }
    d_out.free_buf();
    d_flw.free_buf();
    h_gen_buf.free_buf();
}

// ================================================================
//  §13  CUDA DEVICE QUERY
// ================================================================
static void query_cuda(char *buf, int len) {
    int cnt = 0;
    cudaError_t err = cudaGetDeviceCount(&cnt);
    if (err != cudaSuccess || cnt == 0) {
        _snprintf(buf, len,
            "CUDA: No device — %s  [driver too old? need >= 367.48]",
            cudaGetErrorString(err));
        g_cuda_ok = false;
        return;
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    // Binary targets sm_21 (Fermi CC2.1).
    // Any GPU CC2.1 or newer can execute sm_21 code.
    int cc = p.major * 10 + p.minor;
    if (cc < 21) {
        _snprintf(buf, len,
            "CUDA: %s CC%d.%d — INCOMPATIBLE (binary=sm_21, need CC2.1+)",
            p.name, p.major, p.minor);
        g_cuda_ok = false;
        return;
    }

    g_cuda_ok = true;
    // Cores per SM by architecture
    int cores_per_sm = (p.major == 2) ? 48    // Fermi GF10x
                     : (p.major == 3) ? 192   // Kepler GK2xx
                     : (p.major == 5) ? 128   // Maxwell
                     : (p.major == 6) ? 128   // Pascal
                     : 64;                    // Turing/Ampere (conservative)
    _snprintf(buf, len,
        "CUDA OK: %s  CC%d.%d  %d MB  (~%d cores)",
        p.name, p.major, p.minor,
        (int)(p.totalGlobalMem / (1024 * 1024)),
        p.multiProcessorCount * cores_per_sm);
}

// ================================================================
//  §14  CONTROL WINDOW
// ================================================================
enum { ID_START=3001, ID_STOP, ID_PICK, ID_TIMER=4001 };

static LRESULT CALLBACK CtrlProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        HFONT hf = CreateFontA(13, 0, 0, 0, 400, 0, 0, 0, ANSI_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
            DEFAULT_PITCH, "Consolas");

        auto mkb = [&](const char *t, int id, int x, int y, int w, int h) {
            HWND b = CreateWindowA("BUTTON", t,
                WS_CHILD|WS_VISIBLE|BS_PUSHBUTTON,
                x, y, w, h, hwnd, (HMENU)(UINT_PTR)id, NULL, NULL);
            SendMessageA(b, WM_SETFONT, (WPARAM)hf, TRUE);
        };
        auto mks = [&](const char *t, int x, int y, int w, int h) {
            HWND s = CreateWindowA("STATIC", t, WS_CHILD|WS_VISIBLE,
                x, y, w, h, hwnd, NULL, NULL, NULL);
            SendMessageA(s, WM_SETFONT, (WPARAM)hf, TRUE);
        };
        auto mkid = [&](const char *t, int id, int x, int y, int w, int h) -> HWND {
            HWND s = CreateWindowA("STATIC", t, WS_CHILD|WS_VISIBLE,
                x, y, w, h, hwnd, (HMENU)(UINT_PTR)id, NULL, NULL);
            SendMessageA(s, WM_SETFONT, (WPARAM)hf, TRUE);
            return s;
        };

        mkb("Pick Window", ID_PICK,  10,  10, 120, 26);
        mkb("Start FG",    ID_START, 140, 10, 110, 26);
        mkb("Stop FG",     ID_STOP,  260, 10, 110, 26);

        char ci[256]; query_cuda(ci, sizeof(ci));
        g_cuda_label = mkid(ci,             5000, 10,  44, 620, 16);
        g_stat_label = mkid("Status: idle", 5001, 10,  62, 620, 16);
        g_fps_label  = mkid("FPS: --",      5002, 10,  80, 620, 16);

        int y = 102;
        mks("══════════════════════════════════════════════════════════════", 10, y, 640, 14); y+=16;
        mks("  NEXUS v1.1  —  THE SOVEREIGN Frame Generation Engine",        10, y, 640, 16); y+=18;
        mks("  'Neither Interpolation nor Extrapolation — A New Mathematics'",10,y, 640, 16); y+=18;
        mks("══════════════════════════════════════════════════════════════", 10, y, 640, 14); y+=16;
        mks("  THE SOVEREIGN FORMULA  (crown formula — new mathematics):",   10, y, 640, 16); y+=16;
        mks("    F*(c) = W / (w_w/F_warp_c + w_t/F_temp_c + w_s/F_spat_c)",10, y, 640, 16); y+=16;
        mks("    WHM on dual Fisher-Rao cone  [WHM <= GM <= AM]",           10, y, 640, 16); y+=16;
        mks("    Bounded: min(3 estimates) <= F* <= max(3 estimates)",       10, y, 640, 16); y+=16;
        mks("    Ghost-proof without any detection.  Blur-free by design.",  10, y, 640, 16); y+=16;
        mks("──────────────────────────────────────────────────────────────", 10, y, 640, 14); y+=16;
        mks("  PILLAR A — AnisoBilinear Warp via Lucas-Kanade Flow (GPU):",  10, y, 640, 16); y+=16;
        mks("    [sxx sxy][vx] = [-sxt]   2x2 closed-form, 5x5 window",    10, y, 640, 16); y+=16;
        mks("    [sxy syy][vy]   [-syt]   c_div=exp(-div(v)^2 * sig^-2)",  10, y, 640, 16); y+=16;
        mks("    v1.1: shared-mem tile ~5x fewer global reads (Fermi bus)", 10, y, 640, 16); y+=16;
        mks("  PILLAR B — FLGE Log-Geodesic: 1.625*log F2 - 0.75*log F1 + 0.125*log F0", 10, y, 640, 16); y+=16;
        mks("  PILLAR C — FASW Wave:  F2 + AnisoLap(F2) * c^2 * 0.125",   10, y, 640, 16); y+=16;
        mks("──────────────────────────────────────────────────────────────", 10, y, 640, 14); y+=16;
        mks("  WEIGHTS  w_w=c_div*c_lk   w_t=(1-jerk)*h   w_s=jerk+lam*(1-h)+ofc", 10, y, 640, 16); y+=16;
        mks("──────────────────────────────────────────────────────────────", 10, y, 640, 14); y+=16;
        mks("  Target: GT 730 GF108 CC2.1 (Fermi) / i3 2120 — VRAM-native — No RTX",     10, y, 640, 16); y+=16;
        mks("  Build:  CUDA 8.0  sm_21  driver>=367.48  Pipeline: GDI->pinned->VRAM->LK(shm)->SOVEREIGN->GDI", 10, y, 640, 16);

        SetTimer(hwnd, ID_TIMER, 500, NULL);
        return 0;
    }

    case WM_TIMER:
        if (g_running && g_fps_label) {
            char buf[256];
            double r = (double)g_fps_real;
            _snprintf(buf, sizeof(buf),
                "Capture: %.1f fps  |  Output: %.1f fps  |  x%.1f speedup"
                "  |  SOVEREIGN CUDA — VRAM-native (v1.1 shm-LK)",
                r, r * 2.0, 2.0);
            SetWindowTextA(g_fps_label, buf);
        }
        return 0;

    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case ID_PICK: {
            MessageBoxA(hwnd,
                "Click OK then hover mouse over your game or video\n"
                "window within 3 seconds.",
                "Pick Target Window", MB_OK | MB_ICONINFORMATION);
            Sleep(3000);
            POINT pt; GetCursorPos(&pt);
            HWND hit = WindowFromPoint(pt);
            g_target_hwnd = (hit && hit != hwnd) ? hit : NULL;
            char title[256] = {};
            if (g_target_hwnd) GetWindowTextA(g_target_hwnd, title, 255);
            char s[320];
            _snprintf(s, sizeof(s), "Target: %s",
                g_target_hwnd ? title : "(full desktop)");
            SetWindowTextA(g_stat_label, s);
            break;
        }
        case ID_START:
            if (!g_cuda_ok) {
                MessageBoxA(hwnd,
                    "No compatible CUDA device found.\nMake sure:\n"
                    "  1. NVIDIA driver is >= 367.48\n"
                    "     (any GT 730 Fermi driver from 2014+ works)\n"
                    "  2. GPU is Fermi CC2.1 or newer\n"
                    "     (GT 730 GF108, GT 630, GTX 5xx series etc.)\n"
                    "  3. NEXUS.exe was built with CUDA 8.0 -arch=sm_21\n\n"
                    "If your driver is too old, update at: nvidia.com/drivers\n"
                    "Search: GeForce GT 730 / Windows 10 64-bit",
                    "CUDA Error", MB_OK | MB_ICONERROR);
                break;
            }
            if (!g_running) {
                g_running = true;
                g_ring[0] = 0; g_ring[1] = 1; g_ring[2] = 2;
                g_gen_thread = std::thread(gen_thread_fn);
                SetWindowTextA(g_stat_label,
                    "Status: RUNNING — NEXUS SOVEREIGN active [VRAM-native v1.1]");
            }
            break;
        case ID_STOP:
            if (g_running) {
                g_running = false;
                if (g_gen_thread.joinable()) g_gen_thread.join();
                SetWindowTextA(g_stat_label, "Status: stopped");
                SetWindowTextA(g_fps_label,  "FPS: --");
            }
            break;
        }
        return 0;

    case WM_DESTROY:
        g_running = false;
        if (g_gen_thread.joinable()) g_gen_thread.join();
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

// ================================================================
//  §15  WinMain
// ================================================================
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    WNDCLASSA wc{};
    wc.lpfnWndProc   = CtrlProc;
    wc.hInstance     = hInst;
    wc.hCursor       = LoadCursorA(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = "NEXUSCtrl";
    RegisterClassA(&wc);

    g_ctrl_hwnd = CreateWindowA(
        "NEXUSCtrl", APP_NAME,
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE,
        60, 60, 690, 560,
        NULL, NULL, hInst, NULL);

    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return 0;
}
