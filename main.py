import numpy as np
from PIL import Image
import cv2
import json

def render_ellipse_cv2(buffer, x, y, sx, sy, angle, color):
    h, w = buffer.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(round(x)), int(round(y)))
    axes = (max(1, int(round(sx))), max(1, int(round(sy))))
    cv2.ellipse(mask, center, axes, np.degrees(angle), 0, 360, 255, -1)
    where = mask > 0
    for c in range(3):
        buffer[:, :, c][where] += color[c]

def score_candidates_downsampled(residual_small, scale, xs, ys, sxs, sys_, angles):
    h, w = residual_small.shape[:2]
    n = len(xs)
    scores = np.zeros(n)
    colors = np.zeros((n, 3))
    xs_s = xs / scale
    ys_s = ys / scale
    sxs_s = sxs / scale
    sys_s = sys_ / scale
    for j in range(n):
        sx_s = max(1, sxs_s[j])
        sy_s = max(1, sys_s[j])
        radius = max(sx_s, sy_s)
        x0 = max(0, int(xs_s[j] - radius - 1))
        x1 = min(w, int(xs_s[j] + radius + 1))
        y0 = max(0, int(ys_s[j] - radius - 1))
        y1 = min(h, int(ys_s[j] + radius + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        dx = gx - xs_s[j]
        dy = gy - ys_s[j]
        cos_a = np.cos(angles[j])
        sin_a = np.sin(angles[j])
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        mask = ((rx / sx_s) ** 2 + (ry / sy_s) ** 2) <= 1.0
        count = np.sum(mask)
        if count < 1:
            continue
        patch = residual_small[y0:y1, x0:x1]
        col = np.zeros(3)
        for c in range(3):
            col[c] = np.sum(mask * patch[:, :, c]) / count
        colors[j] = col
        scores[j] = count * np.sum(col ** 2)
    return scores, colors

# --- Config ---
MAX_PRIMITIVES = 500
CANDIDATES_PER_ITER = 1000
MIN_SIGMA = 2
MAX_SIGMA_FACTOR = 0.5
DOWNSAMPLE = 4

# Load image
img = Image.open("input.png").convert("RGB")
target = np.array(img, dtype=np.float64)
h, w, _ = target.shape
print(f"Loaded: {w}x{h}")

# Initialize canvas to fuzzy median background
bg_color = np.median(target.reshape(-1, 3), axis=0)
canvas = np.ones_like(target) * bg_color
residual = target - canvas
print(f"Background: [{bg_color[0]:.0f}, {bg_color[1]:.0f}, {bg_color[2]:.0f}]")

max_sigma = max(w, h) * MAX_SIGMA_FACTOR
primitives = []

for i in range(MAX_PRIMITIVES):
    res_small = cv2.resize(residual, (w // DOWNSAMPLE, h // DOWNSAMPLE),
                           interpolation=cv2.INTER_AREA)

    xs = np.random.uniform(0, w, CANDIDATES_PER_ITER)
    ys = np.random.uniform(0, h, CANDIDATES_PER_ITER)
    sxs = np.random.uniform(MIN_SIGMA, max_sigma, CANDIDATES_PER_ITER)
    sys_ = np.random.uniform(MIN_SIGMA, max_sigma, CANDIDATES_PER_ITER)
    angles = np.random.uniform(0, np.pi, CANDIDATES_PER_ITER)

    scores, colors = score_candidates_downsampled(
        res_small, DOWNSAMPLE, xs, ys, sxs, sys_, angles
    )

    best = np.argmax(scores)
    if scores[best] <= 0:
        break

    bx, by, bsx, bsy, ba = xs[best], ys[best], sxs[best], sys_[best], angles[best]
    best_color = colors[best]

    render_ellipse_cv2(canvas, bx, by, bsx, bsy, ba, best_color)
    render_ellipse_cv2(residual, bx, by, bsx, bsy, ba, -best_color)

    primitives.append({
        "x": round(float(bx), 2),
        "y": round(float(by), 2),
        "sx": round(float(bsx), 2),
        "sy": round(float(bsy), 2),
        "angle": round(float(ba), 4),
        "color": [round(float(c), 2) for c in best_color]
    })

    error = np.mean(residual ** 2)
    display = np.clip(canvas, 0, 255).astype(np.uint8)
    display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    cv2.imshow("Reconstruction", display)
    cv2.waitKey(1)
    print(f"Primitive {i+1}/{MAX_PRIMITIVES}, error: {error:.1f}")

cv2.destroyAllWindows()

output = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
output.save("output.png")
print(f"Saved output.png with {len(primitives)} primitives")

with open("primitives.json", "w") as f:
    json.dump({
        "width": w, "height": h,
        "background": [round(float(c), 2) for c in bg_color],
        "primitives": primitives
    }, f, indent=2)
print("Saved primitives.json")