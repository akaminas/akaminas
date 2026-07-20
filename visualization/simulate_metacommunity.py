"""
Spatial trait-structured metacommunity crossing a stress-driven regime shift.


The model couples two fields on a periodic lattice (a torus):

    B(x, y, t)   local biomass / occupancy density
    z(x, y, t)   local community-mean trait

and two static, spatially autocorrelated environmental layers:

    E(x, y)      local trait optimum   (environmental filtering)
    K(x, y)      local carrying capacity (productivity / refugium structure)

Dynamics (see visualization/README.md for the full technical note):

    dB/dt = r B (1 - B/K) M(z,E)  -  S(t) B^2 / (B^2 + h^2)  -  m B
            + D lap(B) + sqrt(B) dW

    d(Bz)/dt = z * [growth and loss terms]              (offspring inherit trait)
             + B * V (E - z) / w^2                      (directional selection)
             + D lap(Bz)                                (trait flux under dispersal)
             + mutational / drift noise

    M(z,E) = exp(-(z - E)^2 / (2 w^2))                  (environmental filtering)

S(t) is a slow, strictly periodic stress driver. Because the Holling type III
loss term makes the system bistable, S(t) carries the landscape across a fold
bifurcation: a patchy collapse to a degraded state, then hysteretic recovery
by dispersal out of high-K refugia.

 LOOPING
----------------
Every stochastic input (the noise field, the disturbance schedule) is
pre-generated for exactly one loop period and replayed identically on every
pass. The system is therefore a deterministic dynamical system under strictly
periodic forcing; because it is dissipative it converges onto a periodic orbit.
We integrate several burn-in loops and record only the last one, so the final
frame flows into the first with no visible seam. The residual state mismatch is
reported at the end of the run.

Author: Alexandros Kaminas
Code licensed under Apache-2.0 (see repository LICENSE).
Rendered animation licensed CC BY 4.0 (see visualization/README.md).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 20260419

NX, NY = 450, 166          # lattice cells (torus)
FRAMES = 96                # frames in one loop
SUBSTEPS = 6               # integration steps per frame
DT = 0.25                  # time step
BURN_LOOPS = 4             # loops integrated before recording the final one

# Ecological parameters
R_GROWTH = 1.00            # intrinsic growth rate
MORTALITY = 0.045          # density-independent loss
H_HALF = 0.14              # half-saturation of the type III loss term
W_NICHE = 0.22             # niche width of the environmental filter
V_TRAIT = 0.022            # heritable trait variance (selection response)
D_DISP = 0.065             # dispersal (diffusion) coefficient
SIGMA_B = 0.055            # demographic noise amplitude
SIGMA_Z = 0.016            # trait drift / mutation amplitude

STRESS_MIN, STRESS_MAX = 0.035, 0.250   # range of the periodic driver S(t)

N_DISTURBANCE = 7          # discrete disturbance events per loop

# Rendering
BG = np.array([0.043, 0.055, 0.086])     # #0b0e16 - neutral near-black
B_REF = 0.80                             # biomass mapped to full opacity
CMAP_LO, CMAP_HI = 0.22, 1.00            # trimmed viridis range

FONT_DIR = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    global FONT_DIR
    if FONT_DIR is None:
        import matplotlib
        FONT_DIR = os.path.join(
            os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"
        )
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def laplacian(a: np.ndarray) -> np.ndarray:
    """Five-point Laplacian with periodic boundaries."""
    return (
        np.roll(a, 1, 0) + np.roll(a, -1, 0)
        + np.roll(a, 1, 1) + np.roll(a, -1, 1)
        - 4.0 * a
    )


def correlated_field(rng, ny, nx, scale, seed_shape=None):
    """Spatially autocorrelated field on a torus, normalised to [0, 1]."""
    raw = rng.standard_normal((ny, nx)) if seed_shape is None else seed_shape
    f = gaussian_filter(raw, sigma=scale, mode="wrap")
    f -= f.min()
    if f.max() > 0:
        f /= f.max()
    return f.astype(np.float32)


def viridis_lut(n=512):
    """Trimmed viridis lookup table as an (n, 3) float array in [0, 1]."""
    from matplotlib import colormaps
    cm = colormaps["viridis"]
    t = np.linspace(CMAP_LO, CMAP_HI, n)
    return np.asarray(cm(t))[:, :3]


def stress_driver(phase: np.ndarray | float) -> np.ndarray | float:
    """Strictly periodic environmental stress. phase in [0, 1)."""
    mid = 0.5 * (STRESS_MIN + STRESS_MAX)
    amp = 0.5 * (STRESS_MAX - STRESS_MIN)
    return mid - amp * np.cos(2.0 * np.pi * phase)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class Landscape:
    E: np.ndarray          # trait optimum
    K: np.ndarray          # carrying capacity
    noise: np.ndarray      # (FRAMES, ny_c, nx_c) periodic coarse noise
    disturbances: list     # (frame, cy, cx, radius, strength)


def build_landscape(rng, frames) -> Landscape:
    E = correlated_field(rng, NY, NX, scale=6.5)
    E = 0.06 + 0.88 * E

    K = correlated_field(rng, NY, NX, scale=4.5)
    K = 0.66 + 0.52 * K

    # Periodic coarse noise: white noise smoothed circularly in x, y AND t,
    # so the sequence wraps seamlessly at the loop boundary.
    cy, cx = NY // 4, NX // 4
    raw = rng.standard_normal((frames, cy, cx)).astype(np.float32)
    noise = gaussian_filter(raw, sigma=(1.6, 1.3, 1.3), mode="wrap")
    noise /= noise.std()

    dist = []
    for _ in range(N_DISTURBANCE):
        dist.append((
            int(rng.integers(0, frames)),
            float(rng.uniform(0, NY)),
            float(rng.uniform(0, NX)),
            float(rng.uniform(10, 26)),
            float(rng.uniform(0.55, 0.90)),
        ))
    return Landscape(E, K, noise, dist)


def disturbance_mask(cy, cx, radius, strength):
    """Soft-edged circular knockdown on the torus."""
    yy = np.arange(NY)[:, None]
    xx = np.arange(NX)[None, :]
    dy = np.minimum(np.abs(yy - cy), NY - np.abs(yy - cy))
    dx = np.minimum(np.abs(xx - cx), NX - np.abs(xx - cx))
    d = np.sqrt(dy ** 2 + dx ** 2)
    m = 1.0 - strength * np.exp(-(d / radius) ** 4)
    return m.astype(np.float32)


def simulate(frames, burn_loops, verbose=True):
    rng = np.random.default_rng(SEED)
    land = build_landscape(rng, frames)

    dist_by_frame = {}
    for (f, cy, cx, rad, st) in land.disturbances:
        dist_by_frame.setdefault(f, []).append(disturbance_mask(cy, cx, rad, st))

    # Initial condition: near the vegetated branch, traits locally adapted.
    B = (0.72 * land.K).astype(np.float32)
    z = land.E.copy()
    P = (B * z).astype(np.float32)

    eps = np.float32(1e-3)
    rec_B = np.empty((frames, NY, NX), dtype=np.float32)
    rec_z = np.empty((frames, NY, NX), dtype=np.float32)
    rec_S = np.empty(frames, dtype=np.float32)

    state_at_loop_start = None

    for loop in range(burn_loops + 1):
        recording = (loop == burn_loops)
        if recording:
            state_at_loop_start = (B.copy(), P.copy())

        for f in range(frames):
            phase = f / frames
            S = np.float32(stress_driver(phase))

            # Upsample the periodic coarse noise for this frame.
            xi = zoom(land.noise[f], (NY / land.noise.shape[1],
                                      NX / land.noise.shape[2]), order=1)
            xi = xi[:NY, :NX].astype(np.float32)
            if xi.shape != (NY, NX):
                pad_y, pad_x = NY - xi.shape[0], NX - xi.shape[1]
                xi = np.pad(xi, ((0, pad_y), (0, pad_x)), mode="wrap")

            for _ in range(SUBSTEPS):
                z = P / (B + eps)
                np.clip(z, -0.25, 1.25, out=z)

                # Environmental filtering
                M = np.exp(-((z - land.E) ** 2) / (2.0 * W_NICHE ** 2))

                growth = R_GROWTH * B * (1.0 - B / land.K) * M
                loss_stress = S * B * B / (B * B + H_HALF ** 2)
                loss_base = MORTALITY * B
                net = growth - loss_stress - loss_base

                lapB = laplacian(B)
                lapP = laplacian(P)

                demo = SIGMA_B * np.sqrt(np.maximum(B, 0.0)) * xi

                dB = (net + D_DISP * lapB) * DT + demo * np.sqrt(DT)

                # Trait moment: offspring inherit z; selection pulls z toward E.
                selection = B * V_TRAIT * (land.E - z) / (W_NICHE ** 2)
                dP = (z * net + selection + D_DISP * lapP) * DT \
                     + (z * demo + SIGMA_Z * np.sqrt(np.maximum(B, 0.0)) * xi) \
                     * np.sqrt(DT)

                B = np.clip(B + dB, 0.0, 1.4).astype(np.float32)
                P = (P + dP).astype(np.float32)
                np.clip(P, -0.3 * (B + eps), 1.3 * (B + eps), out=P)

            for m in dist_by_frame.get(f, []):
                B *= m
                P *= m

            if recording:
                rec_B[f] = B
                rec_z[f] = P / (B + eps)
                rec_S[f] = S

        if verbose:
            cover = float((B > 0.15).mean())
            print(f"  loop {loop + 1}/{burn_loops + 1}  end cover={cover:.3f}  "
                  f"meanB={float(B.mean()):.3f}")

    # Seam diagnostic: how far did the recorded loop drift from its start?
    B0, P0 = state_at_loop_start
    drift = float(np.abs(B - B0).mean() / max(B0.mean(), 1e-6))
    if verbose:
        cov = (rec_B > 0.15).mean(axis=(1, 2))
        print(f"  seam drift (relative mean |dB|): {drift:.4%}")
        print(f"  cover range over loop: {cov.min():.3f} -> {cov.max():.3f}")

    return rec_B, rec_z, rec_S, land, drift


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_field(Bf, zf, lut):
    """Composite one frame of the lattice into an RGB float array."""
    zn = np.clip(zf, 0.0, 1.0)
    idx = (zn * (lut.shape[0] - 1)).astype(np.int32)
    rgb = lut[idx]                                   # (NY, NX, 3)

    alpha = np.clip(Bf / B_REF, 0.0, 1.0) ** 0.72
    img = BG[None, None, :] + (rgb - BG[None, None, :]) * alpha[..., None]

    # Restrained bloom around dense, high-trait patches.
    lum = (alpha * np.clip((zn - 0.35) / 0.65, 0.0, 1.0)).astype(np.float32)
    glow = gaussian_filter(lum, sigma=2.4)
    img = img + 0.16 * (glow[..., None] * rgb)

    return np.clip(img, 0.0, 1.0)


def draw_strip(img_pil, scale, lut, rec_S, frame_idx, frames, field_h):
    """Legend strip: trait colour bar and the periodic stress driver."""
    d = ImageDraw.Draw(img_pil)
    W = img_pil.width
    s = scale / 2.0
    strip_top = field_h

    muted = (128, 141, 166)
    faint = (74, 84, 105)

    f_small = _font(max(9, int(11.5 * s)))
    f_tiny = _font(max(8, int(10 * s)))

    # --- separator ---
    d.line([(0, strip_top), (W, strip_top)], fill=(26, 32, 46), width=max(1, int(s)))

    pad = int(22 * s)
    base_y = strip_top + int(26 * s)

    # --- trait colour bar (left) ---
    bar_w, bar_h = int(150 * s), int(8 * s)
    d.text((pad, strip_top + int(11 * s)), "community mean trait",
           font=f_small, fill=muted)
    for i in range(bar_w):
        c = lut[int(i / max(bar_w - 1, 1) * (lut.shape[0] - 1))]
        col = tuple(int(255 * v) for v in c)
        d.line([(pad + i, base_y), (pad + i, base_y + bar_h)], fill=col)
    d.text((pad, base_y + bar_h + int(4 * s)), "low", font=f_tiny, fill=faint)
    tw = d.textlength("high", font=f_tiny)
    d.text((pad + bar_w - tw, base_y + bar_h + int(4 * s)), "high",
           font=f_tiny, fill=faint)

    # --- brightness key (middle) ---
    mid_x = pad + bar_w + int(56 * s)
    d.text((mid_x, strip_top + int(11 * s)), "biomass density",
           font=f_small, fill=muted)
    for i in range(bar_w):
        a = (i / max(bar_w - 1, 1)) ** 0.72
        c = BG + (np.array([0.86, 0.88, 0.92]) - BG) * a
        col = tuple(int(255 * v) for v in c)
        d.line([(mid_x + i, base_y), (mid_x + i, base_y + bar_h)], fill=col)
    d.text((mid_x, base_y + bar_h + int(4 * s)), "bare", font=f_tiny, fill=faint)
    tw = d.textlength("dense", font=f_tiny)
    d.text((mid_x + bar_w - tw, base_y + bar_h + int(4 * s)), "dense",
           font=f_tiny, fill=faint)

    # --- stress driver sparkline (right) ---
    spark_w, spark_h = int(180 * s), int(26 * s)
    sx = W - pad - spark_w
    sy = strip_top + int(20 * s)
    d.text((sx, strip_top + int(5 * s)), "environmental stress S(t)",
           font=f_small, fill=muted)

    Smin, Smax = float(rec_S.min()), float(rec_S.max())
    span = max(Smax - Smin, 1e-9)
    pts = []
    for i in range(frames):
        px = sx + spark_w * i / (frames - 1)
        py = sy + spark_h * (1.0 - (rec_S[i] - Smin) / span)
        pts.append((px, py))
    d.line(pts, fill=(58, 68, 88), width=max(1, int(1.5 * s)), joint="curve")

    cx, cy = pts[frame_idx]
    r = max(2, int(2.6 * s))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(253, 231, 137))

    return img_pil


def render_frames(rec_B, rec_z, rec_S, scale, indices=None, verbose=True):
    """Render the given frame indices (default: the whole loop)."""
    lut = viridis_lut()
    frames = rec_B.shape[0]
    if indices is None:
        indices = range(frames)
    field_h = NY * scale
    strip_h = int(68 * scale / 2)
    out = []

    for n, f in enumerate(indices):
        img = render_field(rec_B[f], rec_z[f], lut)
        arr = (img * 255.0 + 0.5).astype(np.uint8)
        pil = Image.fromarray(arr, "RGB").resize(
            (NX * scale, field_h), Image.LANCZOS
        )

        canvas = Image.new("RGB", (NX * scale, field_h + strip_h),
                           tuple(int(255 * v) for v in BG))
        canvas.paste(pil, (0, 0))
        draw_strip(canvas, scale, lut, rec_S, f, frames, field_h)
        out.append(canvas)
        if verbose and (n + 1) % 24 == 0:
            print(f"  rendered {n + 1}")
    return out


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def build_palette(frames_list, colors=250):
    """Global palette sampled across the whole loop."""
    step = max(1, len(frames_list) // 12)
    sample = []
    for im in frames_list[::step]:
        a = np.asarray(im).reshape(-1, 3)
        idx = np.random.default_rng(0).choice(a.shape[0],
                                              size=min(40000, a.shape[0]),
                                              replace=False)
        sample.append(a[idx])
    sample = np.concatenate(sample, axis=0)
    n = int(np.ceil(np.sqrt(sample.shape[0])))
    sample = np.pad(sample, ((0, n * n - sample.shape[0]), (0, 0)), mode="edge")
    rep = Image.fromarray(sample.reshape(n, n, 3), "RGB")
    return rep.quantize(colors=colors, method=Image.Quantize.MAXCOVERAGE)


def save_gif(frames_list, path, duration_ms, colors=200, dither=True):
    """GIF fallback. Note: LZW compresses these continuous fields very poorly
    (~9-16 MB); animated WebP is used for the README instead."""
    pal = build_palette(frames_list, colors=colors)
    d = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    conv = [im.quantize(palette=pal, dither=d) for im in frames_list]
    conv[0].save(
        path, save_all=True, append_images=conv[1:], loop=0,
        duration=duration_ms, disposal=1, optimize=True,
    )
    return os.path.getsize(path)


def save_webp(frames_list, path, duration_ms, quality=85):
    """Animated WebP: full 24-bit colour, ~8x smaller than an equivalent GIF."""
    frames_list[0].save(
        path, save_all=True, append_images=frames_list[1:], loop=0,
        duration=duration_ms, quality=quality, method=6, minimize_size=True,
    )
    return os.path.getsize(path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--burn", type=int, default=BURN_LOOPS)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--outdir", default="../assets")
    ap.add_argument("--name", default="metacommunity-regime-shift")
    ap.add_argument("--still-frame", type=int, default=None,
                    help="frame index for the static fallback (default: auto)")
    ap.add_argument("--quality", type=int, default=85, help="WebP quality")
    ap.add_argument("--also-gif", action="store_true",
                    help="additionally emit a (much larger) GIF")
    ap.add_argument("--no-dither", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.abspath(os.path.join(here, args.outdir))
    os.makedirs(outdir, exist_ok=True)

    print("simulating ...")
    rec_B, rec_z, rec_S, land, drift = simulate(args.frames, args.burn)

    print("rendering ...")
    frames_list = render_frames(rec_B, rec_z, rec_S, args.scale)

    dur = int(round(1000 / args.fps))
    webp_path = os.path.join(outdir, f"{args.name}.webp")
    size = save_webp(frames_list, webp_path, dur, quality=args.quality)
    print(f"wrote {webp_path}  ({size / 1e6:.2f} MB, {len(frames_list)} frames, "
          f"{len(frames_list) / args.fps:.1f} s loop)")

    if args.also_gif:
        gif_path = os.path.join(outdir, f"{args.name}.gif")
        size = save_gif(frames_list, gif_path, dur, dither=not args.no_dither)
        print(f"wrote {gif_path}  ({size / 1e6:.2f} MB)")

    # Static fallback: the frame with the richest spatial structure
    # (highest spatial variance of biomass) unless overridden.
    if args.still_frame is None:
        var = rec_B.reshape(args.frames, -1).var(axis=1)
        still = int(np.argmax(var))
    else:
        still = args.still_frame
    print(f"still frame: {still}")

    still_frames = render_frames(rec_B, rec_z, rec_S, scale=args.scale * 2,
                                 indices=[still], verbose=False)
    png_path = os.path.join(outdir, f"{args.name}.png")
    still_frames[0].save(png_path, optimize=True)
    print(f"wrote {png_path}  ({os.path.getsize(png_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
