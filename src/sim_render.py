"""
Raster compositing and simulation animation for SARsim.

`composite_layers` is the painting step lifted out of the app's static
map renderer, so the animation background is built by the same code that
draws the paper figure. That matters: if the two drifted apart, an agent
could appear to walk through a lake that was only missing from the
animation's background.

The animation shows two things at once: where the replicates are now
(the swarm) and where they have collectively been (a graded coverage
ramp). The second is the useful one operationally -- it is the model's
probability of area, evolving in time, which is the thing the paper
argues a static distance-ring map cannot give you.

The GIF is written with Pillow rather than imageio to avoid adding a
dependency. Frames are quantised against an exact palette built from the
finite colour set a frame can contain, so there is no dithering and no
frame-to-frame palette flicker.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

# Coverage ramp, pale to deep. Warm reds are chosen because neither the
# base nor the land-cover style list contains one, so coverage can never
# be confused with terrain.
COVERAGE_RAMP = [
    (255, 228, 225),
    (255, 190, 178),
    (250, 145, 120),
    (231, 84, 58),
    (150, 20, 30),
]

# Fractions of the final peak occupancy at which each ramp level starts.
# Thresholds are taken against the *final* frame so the ramp means the
# same thing throughout the animation and coverage visibly grows.
COVERAGE_LEVELS = (0.01, 0.04, 0.12, 0.32, 0.65)

AGENT_COLOUR = (255, 0, 255)        # current replicate positions
IPP_COLOUR = (0, 0, 0)              # initial planning point
IPP_HALO = (255, 255, 255)
BANNER_COLOUR = (255, 255, 255)
TEXT_COLOUR = (17, 24, 39)

# Target longest edge of an animation frame, in pixels. Large enough to
# read terrain, small enough that a hundred frames fit in memory.
DEFAULT_FRAME_PX = 700

BANNER_H = 30

# Axis furniture. The left margin has to hold the widest tick label,
# which is a negative value in km, plus the rotated axis title.
AXIS_LEFT = 62
AXIS_BOTTOM = 40
# The final x tick label is centred on the right edge of the map, so
# without a margin its outer half is cut off by the image boundary.
AXIS_RIGHT = 26
AXIS_COLOUR = (17, 24, 39)
TICK_LEN = 4


def nice_tick_step(span, target_ticks=6):
    """
    A round tick interval close to span / target_ticks.

    Ticks land on 1, 2, 2.5 or 5 times a power of ten, so a 3 km map is
    labelled every 500 m rather than every 483 m.

    The candidate whose tick count lands nearest the target wins, rather
    than the first one large enough. Taking the first would round a
    wanted interval of 1.1 km up to 2 km and leave a 6.7 km map with
    three labels on it. Ties go to the coarser step, which gives fewer
    and cleaner labels.
    """
    if span <= 0:
        return 1.0
    raw = span / float(max(1, target_ticks))
    magnitude = 10.0 ** math.floor(math.log10(raw))
    candidates = [m * magnitude for m in (1.0, 2.0, 2.5, 5.0, 10.0)]
    return min(
        candidates,
        key=lambda step: (abs(span / step - target_ticks), -step),
    )


def axis_units(half_extent_m):
    """
    Pick metres or kilometres, whichever keeps the labels short.

    Returns (divisor, suffix, decimals).
    """
    if half_extent_m >= 2000:
        return 1000.0, "km", 1
    return 1.0, "m", 0


def tick_values(half_extent_m, step):
    """Tick positions from the centre outwards, symmetric and including 0."""
    count = int(math.floor(half_extent_m / step))
    return [i * step for i in range(-count, count + 1)]


def composite_layers(layer_style, layers, gradient_mask):
    """
    Paint the raster stack into an RGB image.

    Layers are drawn in list order, so later entries cover earlier ones.
    Linear layers are dilated first: a one-cell-wide rasterised line all
    but disappears once a multi-thousand-cell grid is downsampled to a
    normal image, and 1-2 cells of dilation restores visibility without
    meaningfully thickening the feature at full resolution.

    Returns (img, legend_entries), where legend_entries lists
    (label, colour) for layers that were actually drawn and carry a
    label. A None label draws without a legend entry, which is how two
    extraction layers share one visual meaning.
    """
    shape = gradient_mask.shape
    img = np.full((shape[0], shape[1], 3), 255, dtype=np.uint8)
    legend_entries = []

    for key, label, color, dilate_iters, is_fill in layer_style:
        mask = gradient_mask if key == "gradient" else layers.get(key)
        if mask is None:
            continue
        mask = (mask == 1)
        if not is_fill and dilate_iters > 0:
            mask = binary_dilation(mask, iterations=dilate_iters)
        if not mask.any():
            continue
        img[mask] = color
        if label is not None:
            legend_entries.append((label, color))

    return img, legend_entries


def _pad_to_multiple(arr, factor, fill):
    """Pad the trailing edges so the array divides evenly into blocks."""
    h, w = arr.shape[:2]
    ph = (-h) % factor
    pw = (-w) % factor
    if ph == 0 and pw == 0:
        return arr
    pad = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad, mode="constant", constant_values=fill)


def downsample_darkest(img, factor):
    """
    Shrink an RGB image by keeping the darkest pixel in each block.

    Area averaging fades thin lines into the white background, which is
    fatal here: trails and streams are one cell wide and are exactly what
    the operator needs in order to see agents following them. Every
    feature colour is darker than the white background, so darkest-wins
    preserves them. It also keeps the output colour set finite, which is
    what lets the GIF palette be exact.
    """
    if factor <= 1:
        return img
    padded = _pad_to_multiple(img, factor, 255)
    h, w = padded.shape[:2]
    blocks = padded.reshape(h // factor, factor, w // factor, factor, 3)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(
        h // factor, w // factor, factor * factor, 3
    )
    lum = blocks.astype(np.int32) @ np.array([299, 587, 114], dtype=np.int32)
    pick = np.argmin(lum, axis=2)
    ii, jj = np.indices(pick.shape)
    return blocks[ii, jj, pick]


def _stamp(frame, rows, cols, colour, radius):
    """Draw square markers of the given radius at the listed cells."""
    fh, fw = frame.shape[:2]
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            frame[np.clip(rows + dr, 0, fh - 1), np.clip(cols + dc, 0, fw - 1)] = colour


def _load_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow < 10.1 has no size argument on the default font.
        return ImageFont.load_default()


def _build_palette(layer_colours):
    """
    An exact palette for the finite colour set the frames can contain.

    Because downsampling picks whole pixels rather than blending them,
    every colour in a frame is a layer colour, white, or one of the
    overlay colours. Quantising against that list means no dithering and
    no per-frame palette recomputation. The greys are only there to give
    antialiased text somewhere sensible to land instead of snapping to a
    terrain colour.
    """
    palette = [(255, 255, 255)]
    extras = list(COVERAGE_RAMP) + [AGENT_COLOUR, IPP_COLOUR, TEXT_COLOUR]
    for colour in list(layer_colours) + extras:
        if tuple(colour) not in palette:
            palette.append(tuple(colour))
    for grey in (32, 64, 96, 128, 160, 192, 224):
        if (grey, grey, grey) not in palette:
            palette.append((grey, grey, grey))
    palette = palette[:256]

    flat = []
    for colour in palette:
        flat.extend(colour)
    flat.extend([0, 0, 0] * (256 - len(palette)))

    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat)
    return pal_img


SCALE_BOX = 11
SCALE_GAP = 3
SCALE_LABEL = "coverage"


def _scale_width(draw, font):
    label_w = draw.textlength(SCALE_LABEL, font=font)
    return int(label_w) + 8 + len(COVERAGE_RAMP) * (SCALE_BOX + SCALE_GAP)


def _draw_scale(draw, x, y, font):
    """Coverage ramp key, drawn at the right of the banner."""
    draw.text((x, y - 1), SCALE_LABEL, fill=TEXT_COLOUR, font=font)
    x += int(draw.textlength(SCALE_LABEL, font=font)) + 8
    for colour in COVERAGE_RAMP:
        draw.rectangle([x, y - 1, x + SCALE_BOX, y + SCALE_BOX - 1], fill=colour)
        x += SCALE_BOX + SCALE_GAP


def _fit_label(draw, font, segments, max_width):
    """
    Drop trailing segments until the banner label fits.

    Narrow frames come from small grids, and on those the caption and the
    clock matter more than the cell size, so segments are ordered most
    important first and truncated from the end.
    """
    for n in range(len(segments), 0, -1):
        text = "    ".join(segments[:n])
        if draw.textlength(text, font=font) <= max_width:
            return text
    return segments[0]


def _rotated_text(text, font, colour=AXIS_COLOUR, background=BANNER_COLOUR):
    """Render a caption turned through 90 degrees, for the y axis title."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = int(probe.textlength(text, font=font)) + 4
    height = 16
    strip = Image.new("RGB", (width, height), background)
    ImageDraw.Draw(strip).text((2, 0), text, fill=colour, font=font)
    return strip.rotate(90, expand=True)


def _build_axis_chrome(fw, fh, half_x_m, half_y_m, font):
    """
    Draw the frame, ticks and axis titles once.

    The furniture is identical in every frame, so it is rendered into a
    template that each frame is stamped into. That keeps the per-frame
    work to copying an array rather than re-drawing text sixty times.
    Distances are measured from the centre of the extent, which is where
    the extraction places the IPP.
    """
    total_w = AXIS_LEFT + fw + AXIS_RIGHT
    total_h = BANNER_H + fh + AXIS_BOTTOM
    chrome = Image.new("RGB", (total_w, total_h), BANNER_COLOUR)
    draw = ImageDraw.Draw(chrome)

    left, top = AXIS_LEFT, BANNER_H
    right, bottom = left + fw - 1, top + fh - 1
    draw.rectangle([left - 1, top - 1, right + 1, bottom + 1], outline=AXIS_COLOUR)

    div_x, unit_x, dp_x = axis_units(half_x_m)
    div_y, unit_y, dp_y = axis_units(half_y_m)

    for value in tick_values(half_x_m, nice_tick_step(2 * half_x_m)):
        x = left + int(round((value + half_x_m) / (2 * half_x_m) * (fw - 1)))
        draw.line([x, bottom + 1, x, bottom + 1 + TICK_LEN], fill=AXIS_COLOUR)
        label = f"{value / div_x:.{dp_x}f}"
        width = draw.textlength(label, font=font)
        draw.text((x - width / 2, bottom + TICK_LEN + 4), label,
                  fill=AXIS_COLOUR, font=font)

    for value in tick_values(half_y_m, nice_tick_step(2 * half_y_m)):
        # Row 0 is the top of the raster, which is the highest northing,
        # so a positive value sits above the centre line.
        y = top + int(round((half_y_m - value) / (2 * half_y_m) * (fh - 1)))
        draw.line([left - 1 - TICK_LEN, y, left - 1, y], fill=AXIS_COLOUR)
        label = f"{value / div_y:.{dp_y}f}"
        width = draw.textlength(label, font=font)
        draw.text((left - TICK_LEN - 6 - width, y - 7), label,
                  fill=AXIS_COLOUR, font=font)

    x_title = f"East of centre ({unit_x})"
    width = draw.textlength(x_title, font=font)
    draw.text((left + (fw - width) / 2, total_h - 15), x_title,
              fill=AXIS_COLOUR, font=font)

    y_title = _rotated_text(f"North of centre ({unit_y})", font)
    chrome.paste(y_title, (2, top + max(0, (fh - y_title.height) // 2)))

    return np.array(chrome), (left, top)


def render_simulation_gif(
    base_img,
    result,
    output_path,
    layer_colours=(),
    feature_overlay=None,
    frame_px=DEFAULT_FRAME_PX,
    frame_ms=120,
    caption=None,
):
    """
    Animate a simulation result over the static map.

    Parameters
    ----------
    base_img : (H, W, 3) uint8
        The composited terrain, at full grid resolution.
    result : dict
        Output of lp_model.simulate.
    layer_colours : iterable of RGB tuples
        Colours present in `base_img`, used to build the exact palette.
    feature_overlay : (H, W) bool array or None
        Full-resolution mask of features to repaint on top of the
        coverage ramp, typically the structural linear layers.

    Returns a dict describing what was written.
    """
    frames = result["frames"]
    density_frames = result["density_frames"]
    if not frames:
        raise ValueError("Simulation produced no frames to animate")

    h, w = base_img.shape[:2]
    factor = max(1, int(np.ceil(max(h, w) / float(frame_px))))

    small = downsample_darkest(base_img, factor)

    # A small extent (a 1.5 km radius at 10 m/cell is only 300 cells)
    # would otherwise animate at 300 px, too cramped to read and too
    # narrow for the banner. Nearest-neighbour upscaling by a whole
    # number keeps every pixel an exact palette colour, which blending
    # would not.
    scale = max(1, int(frame_px // max(small.shape[:2])))
    if scale > 1:
        small = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)

    fh, fw = small.shape[:2]

    # Features named by the caller are repainted on top of the coverage
    # ramp, because the deeper ramp levels would otherwise swallow the
    # trails and watercourses, and seeing which of them the replicates
    # follow is most of the point of watching the animation.
    #
    # The caller decides what goes on top rather than this function
    # guessing from pixel colour: the elevation-gradient layer is dense
    # in steep terrain and would blanket the map, whereas roads and
    # trails are genuinely sparse. Block-max is used to shrink the mask
    # so a one-cell line survives downsampling.
    feature_mask = None
    if feature_overlay is not None:
        if factor > 1:
            padded = _pad_to_multiple(feature_overlay.astype(bool), factor, False)
            ph, pw = padded.shape
            feature_mask = padded.reshape(
                ph // factor, factor, pw // factor, factor
            ).max(axis=(1, 3))
        else:
            feature_mask = feature_overlay.astype(bool)
        if scale > 1:
            feature_mask = np.repeat(
                np.repeat(feature_mask, scale, axis=0), scale, axis=1
            )
        feature_mask = feature_mask[:fh, :fw]

    # Map each frame pixel to its density bin once, rather than
    # upsampling every density frame to full grid resolution.
    dfac = result["density_factor"]
    dh, dw = density_frames[-1].shape
    row_idx = np.minimum(((np.arange(fh) // scale) * factor) // dfac, dh - 1)
    col_idx = np.minimum(((np.arange(fw) // scale) * factor) // dfac, dw - 1)
    take = np.ix_(row_idx, col_idx)

    peak = float(density_frames[-1].max())
    thresholds = [peak * f for f in COVERAGE_LEVELS] if peak > 0 else None

    # One replicate should be a visible dot; five hundred should read as
    # a swarm rather than a solid block, so the marker shrinks as the
    # sample count grows.
    n_samples = frames[0].shape[0]
    agent_radius = 1 if n_samples <= 250 else 0
    if max(fh, fw) >= 600:
        agent_radius += 1
    ipp_radius = agent_radius + 2

    ipp_r, ipp_c = result["ipp_cell"]
    ipp_r = (ipp_r // factor) * scale + scale // 2
    ipp_c = (ipp_c // factor) * scale + scale // 2
    ipp_rows, ipp_cols = np.array([ipp_r]), np.array([ipp_c])

    font = _load_font(14)
    tick_font = _load_font(11)
    palette = _build_palette(layer_colours)
    times = result["frame_times_h"]
    cell_size_m = result["stats"]["cell_size_m"]

    # Axes are measured in ground distance from the centre of the extent,
    # so they stay meaningful whatever the cell size and however far the
    # image was downsampled to fit the frame.
    half_x_m = w * cell_size_m / 2.0
    half_y_m = h * cell_size_m / 2.0
    chrome, (map_x, map_y) = _build_axis_chrome(fw, fh, half_x_m, half_y_m, tick_font)

    pil_frames = []
    for i, positions in enumerate(frames):
        frame = chrome.copy()
        canvas = frame[map_y:map_y + fh, map_x:map_x + fw]
        canvas[:] = small

        # Cumulative coverage, banded rather than blended so that the
        # palette stays exact and terrain shows through the pale levels.
        if thresholds is not None:
            dens = density_frames[i][take]
            for level, colour in zip(thresholds, COVERAGE_RAMP):
                np.copyto(canvas, np.array(colour, dtype=np.uint8),
                          where=(dens >= level)[:, :, None])
            if feature_mask is not None:
                np.copyto(canvas, small, where=feature_mask[:, :, None])

        # Markers are stamped into the canvas view, so their coordinates
        # stay in map space and need no axis offset.
        _stamp(canvas,
               (positions[:, 0] // factor) * scale + scale // 2,
               (positions[:, 1] // factor) * scale + scale // 2,
               AGENT_COLOUR, agent_radius)

        # IPP last, with a halo, so it survives both the swarm and the
        # darkest end of the coverage ramp.
        _stamp(canvas, ipp_rows, ipp_cols, IPP_HALO, ipp_radius + 1)
        _stamp(canvas, ipp_rows, ipp_cols, IPP_COLOUR, ipp_radius)

        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)

        segments = [f"t = {times[i]:5.1f} h", f"{n_samples} replicates",
                    f"{cell_size_m:.1f} m/cell"]
        if caption:
            segments.insert(0, caption)

        banner_w = frame.shape[1]
        scale_w = _scale_width(draw, font)
        show_scale = banner_w > scale_w + 160
        room = banner_w - 16 - (scale_w + 16 if show_scale else 0)
        draw.text((8, 8), _fit_label(draw, font, segments, room),
                  fill=TEXT_COLOUR, font=font)
        if show_scale:
            _draw_scale(draw, banner_w - 8 - scale_w, 9, font)

        pil_frames.append(pil.quantize(palette=palette, dither=Image.Dither.NONE))

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    return {
        "path": str(output_path),
        "frames": len(pil_frames),
        "frame_size": [fh + BANNER_H + AXIS_BOTTOM, fw + AXIS_LEFT + AXIS_RIGHT],
        "map_size": [fh, fw],
        "downsample_factor": factor,
        "upscale_factor": scale,
        "frame_ms": frame_ms,
    }