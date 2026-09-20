from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("D:/EMRDM-main/docs")
MAIN_SVG = ROOT / "emrdm_framework_publication.svg"
MAIN_PNG = ROOT / "emrdm_framework_publication.png"
DET_SVG = ROOT / "emrdm_detection_pipeline_external.svg"
DET_PNG = ROOT / "emrdm_detection_pipeline_external.png"


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


PALETTE = {
    "bg": "#f6f2ea",
    "panel": "#fffdf9",
    "title": "#183b5b",
    "navy": "#224d73",
    "blue": "#5b84c4",
    "blue_fill": "#eef4ff",
    "green": "#4f8668",
    "green_fill": "#eef8f2",
    "orange": "#b96c34",
    "orange_fill": "#fff3ea",
    "purple": "#8b62b8",
    "purple_fill": "#f6efff",
    "gray": "#6b7280",
    "gray_fill": "#f5f6f8",
    "ink": "#1f2933",
    "muted": "#5f6b7a",
    "white": "#ffffff",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/calibrib.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


@dataclass
class BoxSpec:
    x: int
    y: int
    w: int
    h: int
    fill: str
    stroke: str
    lines: Sequence[str]
    font_size: int = 24
    text_color: str = PALETTE["ink"]
    radius: int = 20
    stroke_width: int = 2
    bold: bool = True
    align: str = "center"
    dashed: bool = False


def draw_svg_box(spec: BoxSpec) -> str:
    dash = ' stroke-dasharray="12 9"' if spec.dashed else ""
    out = [
        (
            f'<rect x="{spec.x}" y="{spec.y}" width="{spec.w}" height="{spec.h}" '
            f'rx="{spec.radius}" ry="{spec.radius}" fill="{spec.fill}" '
            f'stroke="{spec.stroke}" stroke-width="{spec.stroke_width}"{dash}/>'
        )
    ]
    line_step = spec.font_size + 7
    total_h = len(spec.lines) * line_step
    start_y = spec.y + spec.h / 2 - total_h / 2 + spec.font_size
    for idx, line in enumerate(spec.lines):
        if spec.align == "left":
            x = spec.x + 20
            anchor = "start"
        else:
            x = spec.x + spec.w / 2
            anchor = "middle"
        weight = "700" if spec.bold else "500"
        out.append(
            f'<text x="{x}" y="{start_y + idx * line_step}" font-size="{spec.font_size}" '
            f'text-anchor="{anchor}" font-family="Arial, Segoe UI, sans-serif" '
            f'fill="{spec.text_color}" font-weight="{weight}">{escape(line)}</text>'
        )
    return "\n".join(out)


def svg_text(x: int, y: int, text: str, font_size: int, color: str, weight: str = "700") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{font_size}" font-family="Arial, Segoe UI, sans-serif" '
        f'fill="{color}" font-weight="{weight}">{escape(text)}</text>'
    )


def svg_poly_arrow(points: Iterable[tuple[int, int]], color: str, width: int = 4, dashed: bool = False) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>'
    )


def svg_badge(x: int, y: int, w: int, h: int, fill: str, stroke: str, text: str) -> str:
    return draw_svg_box(
        BoxSpec(
            x=x,
            y=y,
            w=w,
            h=h,
            fill=fill,
            stroke=stroke,
            lines=[text],
            font_size=20,
            radius=18,
            stroke_width=2,
        )
    )


def draw_png_box(draw: ImageDraw.ImageDraw, spec: BoxSpec) -> None:
    draw.rounded_rectangle(
        (spec.x, spec.y, spec.x + spec.w, spec.y + spec.h),
        radius=spec.radius,
        fill=hex_rgb(spec.fill),
        outline=hex_rgb(spec.stroke),
        width=spec.stroke_width,
    )
    if spec.dashed:
        # Overlay a dashed border for external-tool boxes.
        dash = 14
        gap = 9
        x1, y1, x2, y2 = spec.x, spec.y, spec.x + spec.w, spec.y + spec.h
        for sx in range(x1 + 18, x2 - 18, dash + gap):
            draw.line((sx, y1, min(sx + dash, x2 - 18), y1), fill=hex_rgb(spec.stroke), width=2)
            draw.line((sx, y2, min(sx + dash, x2 - 18), y2), fill=hex_rgb(spec.stroke), width=2)
        for sy in range(y1 + 18, y2 - 18, dash + gap):
            draw.line((x1, sy, x1, min(sy + dash, y2 - 18)), fill=hex_rgb(spec.stroke), width=2)
            draw.line((x2, sy, x2, min(sy + dash, y2 - 18)), fill=hex_rgb(spec.stroke), width=2)

    font = load_font(spec.font_size, bold=spec.bold)
    line_step = spec.font_size + 7
    total_h = len(spec.lines) * line_step
    start_y = spec.y + spec.h / 2 - total_h / 2
    for idx, line in enumerate(spec.lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        if spec.align == "left":
            tx = spec.x + 18
        else:
            tx = spec.x + (spec.w - (bbox[2] - bbox[0])) / 2
        ty = start_y + idx * line_step
        draw.text((tx, ty), line, font=font, fill=hex_rgb(spec.text_color))


def draw_png_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    size: int,
    color: str,
    bold: bool = True,
) -> None:
    draw.text((x, y), text, font=load_font(size, bold=bold), fill=hex_rgb(color))


def draw_png_arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    color: str,
    width: int = 4,
    dashed: bool = False,
) -> None:
    rgb = hex_rgb(color)
    if dashed:
        for idx in range(len(points) - 1):
            x1, y1 = points[idx]
            x2, y2 = points[idx + 1]
            dx = x2 - x1
            dy = y2 - y1
            length = max((dx * dx + dy * dy) ** 0.5, 1.0)
            ux, uy = dx / length, dy / length
            pos = 0.0
            while pos < length:
                end = min(length, pos + 12)
                sx = x1 + ux * pos
                sy = y1 + uy * pos
                ex = x1 + ux * end
                ey = y1 + uy * end
                draw.line((sx, sy, ex, ey), fill=rgb, width=width)
                pos += 20
    else:
        draw.line(points, fill=rgb, width=width)

    x1, y1 = points[-2]
    x2, y2 = points[-1]
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = (x2, y2)
    left = (x2 - 16 * ux + 8 * px, y2 - 16 * uy + 8 * py)
    right = (x2 - 16 * ux - 8 * px, y2 - 16 * uy - 8 * py)
    draw.polygon((tip, left, right), fill=rgb)


def make_svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["navy"]}"/>
  </marker>
</defs>
"""


def render_main_svg() -> str:
    width, height = 2400, 1500
    parts = [make_svg_header(width, height)]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')
    parts.append(draw_svg_box(BoxSpec(40, 28, 2320, 120, PALETTE["title"], PALETTE["title"], [], radius=28, stroke_width=1)))
    parts.append(svg_text(70, 86, "EMRDM: Semantic-Routed Residual Diffusion for Cloud Removal", 44, PALETTE["white"], "700"))
    parts.append(svg_text(70, 122, "Core restoration framework highlighting the proposed innovations", 22, "#d8e7f5", "500"))

    parts.append(draw_svg_box(BoxSpec(40, 180, 2320, 1260, PALETTE["panel"], PALETTE["navy"], [], radius=32, stroke_width=3)))
    parts.append(svg_text(70, 238, "Main Method Figure", 34, PALETTE["navy"], "700"))

    # Input column
    parts.append(draw_svg_box(BoxSpec(90, 300, 320, 970, PALETTE["green_fill"], PALETTE["green"], [], radius=28, stroke_width=3)))
    parts.append(svg_text(120, 350, "Input supervision and conditions", 28, PALETTE["green"], "700"))
    input_boxes = [
        BoxSpec(120, 390, 260, 110, PALETTE["white"], PALETTE["green"], ["Cloudy image", "cond_image / mean input"], 24),
        BoxSpec(120, 540, 260, 92, PALETTE["white"], PALETTE["green"], ["Cloud mask", "M"], 24),
        BoxSpec(120, 670, 260, 92, PALETTE["white"], PALETTE["green"], ["Semantic mask", "semantic_mask"], 24),
        BoxSpec(120, 800, 260, 110, PALETTE["white"], PALETTE["green"], ["Clear target", "label / x0"], 24),
        BoxSpec(120, 960, 260, 120, "#f8fff9", PALETTE["green"], ["Identity first stage", "No latent compression", "x0 and mu stay image-aligned"], 22),
    ]
    for spec in input_boxes:
        parts.append(draw_svg_box(spec))
    for y1, y2 in [(500, 540), (632, 670), (762, 800), (910, 960)]:
        parts.append(svg_poly_arrow([(250, y1), (250, y2)], PALETTE["green"], 3))

    # Training panel
    parts.append(draw_svg_box(BoxSpec(460, 300, 1320, 740, PALETTE["blue_fill"], PALETTE["blue"], [], radius=30, stroke_width=3)))
    parts.append(svg_text(490, 350, "Training graph", 30, PALETTE["blue"], "700"))

    parts.append(draw_svg_box(BoxSpec(500, 390, 320, 120, PALETTE["white"], PALETTE["blue"], ["Residual diffusion engine", "x0 = label", "mu = cond_image"], 21)))
    parts.append(draw_svg_box(BoxSpec(860, 390, 250, 120, PALETTE["white"], PALETTE["blue"], ["EDM sigma-to-state", "st = 1 / (1 + alpha * sigma)"], 20)))
    parts.append(draw_svg_box(BoxSpec(1140, 390, 250, 120, PALETTE["white"], PALETTE["blue"], ["General conditioner", "Identity embedder", "for cond_image"], 20)))
    parts.append(draw_svg_box(BoxSpec(1420, 390, 320, 120, PALETTE["white"], PALETTE["blue"], ["Residual denoiser wrapper", "EDM scaling", "concat conditioning"], 20)))

    core_x, core_y, core_w, core_h = 560, 560, 1080, 360
    parts.append(draw_svg_box(BoxSpec(core_x, core_y, core_w, core_h, PALETTE["white"], PALETTE["navy"], [], radius=28, stroke_width=3)))
    parts.append(svg_text(core_x + 26, core_y + 42, "Semantic-routed dual-branch denoiser", 30, PALETTE["navy"], "700"))
    parts.append(draw_svg_box(BoxSpec(610, 630, 250, 120, PALETTE["green_fill"], PALETTE["green"], ["Route-mask fusion", "semantic_mask U cloud mask", "patch_size = 128"], 22)))
    parts.append(draw_svg_box(BoxSpec(900, 630, 220, 120, PALETTE["gray_fill"], PALETTE["gray"], ["Patch scoring", "route_mode = max", "threshold = 0.005"], 22)))
    parts.append(draw_svg_box(BoxSpec(1160, 620, 210, 150, PALETTE["orange_fill"], PALETTE["orange"], ["Light branch", "Lightweight", "background denoiser"], 22)))
    parts.append(draw_svg_box(BoxSpec(1400, 620, 210, 150, PALETTE["purple_fill"], PALETTE["purple"], ["Heavy branch", "ImageTransformer", "backbone"], 22)))
    parts.append(draw_svg_box(BoxSpec(1080, 810, 420, 82, "#fbfcff", PALETTE["navy"], ["Patch merge and image reconstruction"], 24)))
    parts.append(svg_poly_arrow([(860, 690), (900, 690)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1120, 690), (1160, 690)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1120, 690), (1400, 690)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1265, 770), (1265, 810)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1505, 770), (1505, 810), (1500, 810)], PALETTE["navy"], 4))

    parts.append(draw_svg_box(BoxSpec(1820, 300, 470, 740, PALETTE["orange_fill"], PALETTE["orange"], [], radius=30, stroke_width=3)))
    parts.append(svg_text(1850, 350, "Optimization objectives", 30, PALETTE["orange"], "700"))
    loss_boxes = [
        BoxSpec(1860, 400, 390, 120, PALETTE["white"], PALETTE["orange"], ["Cloud-aware importance loss", "Higher weights on cloud-covered", "foreground pixels"], 20),
        BoxSpec(1860, 560, 390, 140, PALETTE["white"], PALETTE["orange"], ["Asymmetric semantic feature matching", "VGG relu2_2 / relu3_3 / relu4_3", "Foreground and background", "use different weights"], 18),
        BoxSpec(1860, 735, 390, 105, PALETTE["white"], PALETTE["orange"], ["Residual EDM weighting", "Sigma-aware loss scaling during training"], 22),
        BoxSpec(1860, 885, 390, 100, "#fffaf5", PALETTE["orange"], ["Total loss", "pixel reconstruction + 0.10 *", "feature matching"], 20),
    ]
    for spec in loss_boxes:
        parts.append(draw_svg_box(spec))

    # Inference strip
    parts.append(draw_svg_box(BoxSpec(460, 1090, 1830, 250, "#f7fbf8", PALETTE["green"], [], radius=30, stroke_width=3)))
    parts.append(svg_text(490, 1140, "Inference and restoration", 30, PALETTE["green"], "700"))
    infer_boxes = [
        BoxSpec(520, 1185, 310, 105, PALETTE["white"], PALETTE["green"], ["Residual Heun EDM sampler", "Iterative reverse diffusion"], 22),
        BoxSpec(880, 1185, 340, 105, PALETTE["white"], PALETTE["green"], ["Reuse the same semantic-routed denoiser", "semantic_mask, semantic_mask_available", "and M are forwarded"], 18),
        BoxSpec(1270, 1185, 250, 105, PALETTE["white"], PALETTE["green"], ["EMA restoration output", "restored image"], 22),
        BoxSpec(1570, 1185, 430, 105, PALETTE["white"], PALETTE["green"], ["Output can be used for", "visual quality assessment", "or downstream sensing tasks"], 20),
    ]
    for spec in infer_boxes:
        parts.append(draw_svg_box(spec))

    # Innovation badges
    badges = [
        (580, 252, "I1  Route-mask fusion", PALETTE["green_fill"], PALETTE["green"]),
        (850, 252, "I2  Dual-branch routing", PALETTE["blue_fill"], PALETTE["blue"]),
        (1120, 252, "I3  Cloud-aware weighting", PALETTE["orange_fill"], PALETTE["orange"]),
        (1390, 252, "I4  Semantic FM loss", PALETTE["purple_fill"], PALETTE["purple"]),
    ]
    for x, y, text, fill, stroke in badges:
        parts.append(svg_badge(x, y, 240, 42, fill, stroke, text))

    # Main arrows
    parts.append(svg_poly_arrow([(380, 445), (500, 445)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(820, 445), (860, 445)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1110, 445), (1140, 445)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1390, 445), (1420, 445)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1580, 500), (1580, 560)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1740, 445), (1740, 690), (1640, 690)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(380, 720), (560, 720)], PALETTE["green"], 4))
    parts.append(svg_poly_arrow([(380, 855), (500, 855), (500, 445)], PALETTE["green"], 4))
    parts.append(svg_poly_arrow([(1640, 740), (1820, 740), (1820, 460), (1860, 460)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(1500, 852), (1820, 852), (1820, 625), (1860, 625)], PALETTE["navy"], 4))
    parts.append(svg_poly_arrow([(2100, 690), (2100, 735)], PALETTE["orange"], 4))
    parts.append(svg_poly_arrow([(2100, 840), (2100, 885)], PALETTE["orange"], 4))
    parts.append(svg_poly_arrow([(830, 1232), (890, 1232)], PALETTE["green"], 4))
    parts.append(svg_poly_arrow([(1210, 1232), (1270, 1232)], PALETTE["green"], 4))
    parts.append(svg_poly_arrow([(1520, 1232), (1570, 1232)], PALETTE["green"], 4))

    parts.append(svg_text(90, 1400, "Core figure recommendation: keep MMRotate outside this panel and present downstream detection as a separate application figure.", 22, PALETTE["muted"], "500"))
    parts.append("</svg>")
    return "\n".join(parts)


def render_detection_svg() -> str:
    width, height = 2100, 980
    parts = [make_svg_header(width, height)]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')
    parts.append(draw_svg_box(BoxSpec(40, 28, 2020, 120, PALETTE["title"], PALETTE["title"], [], radius=28, stroke_width=1)))
    parts.append(svg_text(70, 86, "Downstream Rotated Object Detection Evaluation", 42, PALETTE["white"], "700"))
    parts.append(svg_text(70, 122, "The detector platform is external to EMRDM and is used only for downstream evaluation", 22, "#d8e7f5", "500"))

    parts.append(draw_svg_box(BoxSpec(40, 180, 2020, 740, PALETTE["panel"], PALETTE["navy"], [], radius=30, stroke_width=3)))
    parts.append(svg_text(70, 238, "Separate application figure", 32, PALETTE["navy"], "700"))

    parts.append(draw_svg_box(BoxSpec(90, 340, 260, 190, PALETTE["green_fill"], PALETTE["green"], ["Cloudy remote-sensing patch", "cloud / test image"], 28)))
    parts.append(draw_svg_box(BoxSpec(430, 300, 420, 270, PALETTE["blue_fill"], PALETTE["blue"], ["EMRDM restoration module", "Semantic-routed residual diffusion", "Produces restored cloud-free imagery"], 28)))
    parts.append(draw_svg_box(BoxSpec(930, 340, 280, 190, PALETTE["white"], PALETTE["blue"], ["Restored image", "for detector input"], 28)))

    parts.append(draw_svg_box(BoxSpec(1290, 260, 600, 360, "#fbfbfd", PALETTE["gray"], [], radius=28, stroke_width=3, dashed=True)))
    parts.append(svg_text(1320, 318, "External detector toolbox", 30, PALETTE["gray"], "700"))
    parts.append(draw_svg_box(BoxSpec(1350, 360, 480, 120, PALETTE["orange_fill"], PALETTE["orange"], ["Off-the-shelf rotated detector", "MMRotate / Faster R-CNN / YOLO-OBB"], 28)))
    parts.append(draw_svg_box(BoxSpec(1350, 520, 480, 70, "#fffaf5", PALETTE["orange"], ["Not a learnable component of EMRDM"], 24)))

    parts.append(draw_svg_box(BoxSpec(930, 680, 320, 120, PALETTE["purple_fill"], PALETTE["purple"], ["Detection outputs", "rotated boxes / category scores"], 26)))
    parts.append(draw_svg_box(BoxSpec(1350, 680, 480, 120, PALETTE["white"], PALETTE["purple"], ["Evaluation metrics", "mAP, AP50, downstream robustness"], 26)))
    parts.append(draw_svg_box(BoxSpec(90, 680, 760, 120, PALETTE["green_fill"], PALETTE["green"], ["Optional annotation conversion and dataset adapters", "for iSAID / DOTA / COCO / OBB evaluation pipelines"], 24)))

    parts.append(svg_badge(470, 250, 340, 44, PALETTE["blue_fill"], PALETTE["blue"], "Main paper method"))
    parts.append(svg_badge(1450, 250, 270, 44, "#f7f7f9", PALETTE["gray"], "External evaluation tool"))

    parts.append(svg_poly_arrow([(350, 435), (430, 435)], PALETTE["navy"], 5))
    parts.append(svg_poly_arrow([(850, 435), (930, 435)], PALETTE["navy"], 5))
    parts.append(svg_poly_arrow([(1210, 435), (1350, 435)], PALETTE["navy"], 5))
    parts.append(svg_poly_arrow([(1590, 480), (1590, 680), (1250, 680), (1250, 740)], PALETTE["orange"], 5))
    parts.append(svg_poly_arrow([(1590, 590), (1590, 680), (1830, 680), (1830, 740)], PALETTE["orange"], 5))
    parts.append(svg_poly_arrow([(850, 740), (930, 740)], PALETTE["green"], 5, dashed=True))

    parts.append(svg_text(90, 875, "Recommended usage in the paper: keep this panel separate from the core EMRDM framework and describe the detector as an external downstream benchmark.", 22, PALETTE["muted"], "500"))
    parts.append("</svg>")
    return "\n".join(parts)


def render_main_png() -> Image.Image:
    width, height = 2400, 1500
    image = Image.new("RGB", (width, height), hex_rgb(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)

    draw_png_box(draw, BoxSpec(40, 28, 2320, 120, PALETTE["title"], PALETTE["title"], [], radius=28, stroke_width=1))
    draw_png_text(draw, 70, 40, "EMRDM: Semantic-Routed Residual Diffusion for Cloud Removal", 44, PALETTE["white"], True)
    draw_png_text(draw, 70, 94, "Core restoration framework highlighting the proposed innovations", 22, "#d8e7f5", False)

    draw_png_box(draw, BoxSpec(40, 180, 2320, 1260, PALETTE["panel"], PALETTE["navy"], [], radius=32, stroke_width=3))
    draw_png_text(draw, 70, 206, "Main Method Figure", 34, PALETTE["navy"], True)

    draw_png_box(draw, BoxSpec(90, 300, 320, 970, PALETTE["green_fill"], PALETTE["green"], [], radius=28, stroke_width=3))
    draw_png_text(draw, 120, 322, "Input supervision and conditions", 28, PALETTE["green"], True)
    for spec in [
        BoxSpec(120, 390, 260, 110, PALETTE["white"], PALETTE["green"], ["Cloudy image", "cond_image / mean input"], 24),
        BoxSpec(120, 540, 260, 92, PALETTE["white"], PALETTE["green"], ["Cloud mask", "M"], 24),
        BoxSpec(120, 670, 260, 92, PALETTE["white"], PALETTE["green"], ["Semantic mask", "semantic_mask"], 24),
        BoxSpec(120, 800, 260, 110, PALETTE["white"], PALETTE["green"], ["Clear target", "label / x0"], 24),
        BoxSpec(120, 960, 260, 120, "#f8fff9", PALETTE["green"], ["Identity first stage", "No latent compression", "x0 and mu stay image-aligned"], 22),
    ]:
        draw_png_box(draw, spec)
    for points in [
        [(250, 500), (250, 540)],
        [(250, 632), (250, 670)],
        [(250, 762), (250, 800)],
        [(250, 910), (250, 960)],
    ]:
        draw_png_arrow(draw, points, PALETTE["green"], 3)

    draw_png_box(draw, BoxSpec(460, 300, 1320, 740, PALETTE["blue_fill"], PALETTE["blue"], [], radius=30, stroke_width=3))
    draw_png_text(draw, 490, 322, "Training graph", 30, PALETTE["blue"], True)
    for spec in [
        BoxSpec(500, 390, 320, 120, PALETTE["white"], PALETTE["blue"], ["Residual diffusion engine", "x0 = label", "mu = cond_image"], 21),
        BoxSpec(860, 390, 250, 120, PALETTE["white"], PALETTE["blue"], ["EDM sigma-to-state", "st = 1 / (1 + alpha * sigma)"], 20),
        BoxSpec(1140, 390, 250, 120, PALETTE["white"], PALETTE["blue"], ["General conditioner", "Identity embedder", "for cond_image"], 20),
        BoxSpec(1420, 390, 320, 120, PALETTE["white"], PALETTE["blue"], ["Residual denoiser wrapper", "EDM scaling", "concat conditioning"], 20),
    ]:
        draw_png_box(draw, spec)

    draw_png_box(draw, BoxSpec(560, 560, 1080, 360, PALETTE["white"], PALETTE["navy"], [], radius=28, stroke_width=3))
    draw_png_text(draw, 586, 582, "Semantic-routed dual-branch denoiser", 30, PALETTE["navy"], True)
    for spec in [
        BoxSpec(610, 630, 250, 120, PALETTE["green_fill"], PALETTE["green"], ["Route-mask fusion", "semantic_mask U cloud mask", "patch_size = 128"], 22),
        BoxSpec(900, 630, 220, 120, PALETTE["gray_fill"], PALETTE["gray"], ["Patch scoring", "route_mode = max", "threshold = 0.005"], 22),
        BoxSpec(1160, 620, 210, 150, PALETTE["orange_fill"], PALETTE["orange"], ["Light branch", "Lightweight", "background denoiser"], 22),
        BoxSpec(1400, 620, 210, 150, PALETTE["purple_fill"], PALETTE["purple"], ["Heavy branch", "ImageTransformer", "backbone"], 22),
        BoxSpec(1080, 810, 420, 82, "#fbfcff", PALETTE["navy"], ["Patch merge and image reconstruction"], 24),
    ]:
        draw_png_box(draw, spec)

    draw_png_box(draw, BoxSpec(1820, 300, 470, 740, PALETTE["orange_fill"], PALETTE["orange"], [], radius=30, stroke_width=3))
    draw_png_text(draw, 1850, 322, "Optimization objectives", 30, PALETTE["orange"], True)
    for spec in [
        BoxSpec(1860, 400, 390, 120, PALETTE["white"], PALETTE["orange"], ["Cloud-aware importance loss", "Higher weights on cloud-covered", "foreground pixels"], 20),
        BoxSpec(1860, 560, 390, 140, PALETTE["white"], PALETTE["orange"], ["Asymmetric semantic feature matching", "VGG relu2_2 / relu3_3 / relu4_3", "Foreground and background", "use different weights"], 18),
        BoxSpec(1860, 735, 390, 105, PALETTE["white"], PALETTE["orange"], ["Residual EDM weighting", "Sigma-aware loss scaling during training"], 22),
        BoxSpec(1860, 885, 390, 100, "#fffaf5", PALETTE["orange"], ["Total loss", "pixel reconstruction + 0.10 *", "feature matching"], 20),
    ]:
        draw_png_box(draw, spec)

    draw_png_box(draw, BoxSpec(460, 1090, 1830, 250, "#f7fbf8", PALETTE["green"], [], radius=30, stroke_width=3))
    draw_png_text(draw, 490, 1114, "Inference and restoration", 30, PALETTE["green"], True)
    for spec in [
        BoxSpec(520, 1185, 310, 105, PALETTE["white"], PALETTE["green"], ["Residual Heun EDM sampler", "Iterative reverse diffusion"], 22),
        BoxSpec(880, 1185, 340, 105, PALETTE["white"], PALETTE["green"], ["Reuse the same semantic-routed denoiser", "semantic_mask, semantic_mask_available", "and M are forwarded"], 18),
        BoxSpec(1270, 1185, 250, 105, PALETTE["white"], PALETTE["green"], ["EMA restoration output", "restored image"], 22),
        BoxSpec(1570, 1185, 430, 105, PALETTE["white"], PALETTE["green"], ["Output can be used for", "visual quality assessment", "or downstream sensing tasks"], 20),
    ]:
        draw_png_box(draw, spec)

    for x, y, text, fill, stroke in [
        (580, 252, "I1  Route-mask fusion", PALETTE["green_fill"], PALETTE["green"]),
        (850, 252, "I2  Dual-branch routing", PALETTE["blue_fill"], PALETTE["blue"]),
        (1120, 252, "I3  Cloud-aware weighting", PALETTE["orange_fill"], PALETTE["orange"]),
        (1390, 252, "I4  Semantic FM loss", PALETTE["purple_fill"], PALETTE["purple"]),
    ]:
        draw_png_box(draw, BoxSpec(x, y, 240, 42, fill, stroke, [text], 18, radius=18, stroke_width=2))

    for pts, color, dashed in [
        ([(380, 445), (500, 445)], PALETTE["navy"], False),
        ([(820, 445), (860, 445)], PALETTE["navy"], False),
        ([(1110, 445), (1140, 445)], PALETTE["navy"], False),
        ([(1390, 445), (1420, 445)], PALETTE["navy"], False),
        ([(1580, 500), (1580, 560)], PALETTE["navy"], False),
        ([(1740, 445), (1740, 690), (1640, 690)], PALETTE["navy"], False),
        ([(380, 720), (560, 720)], PALETTE["green"], False),
        ([(380, 855), (500, 855), (500, 445)], PALETTE["green"], False),
        ([(1640, 740), (1820, 740), (1820, 460), (1860, 460)], PALETTE["navy"], False),
        ([(1500, 852), (1820, 852), (1820, 625), (1860, 625)], PALETTE["navy"], False),
        ([(2100, 690), (2100, 735)], PALETTE["orange"], False),
        ([(2100, 840), (2100, 885)], PALETTE["orange"], False),
        ([(830, 1232), (890, 1232)], PALETTE["green"], False),
        ([(1210, 1232), (1270, 1232)], PALETTE["green"], False),
        ([(1520, 1232), (1570, 1232)], PALETTE["green"], False),
        ([(860, 690), (900, 690)], PALETTE["navy"], False),
        ([(1120, 690), (1160, 690)], PALETTE["navy"], False),
        ([(1120, 690), (1400, 690)], PALETTE["navy"], False),
        ([(1265, 770), (1265, 810)], PALETTE["navy"], False),
        ([(1505, 770), (1505, 810), (1500, 810)], PALETTE["navy"], False),
    ]:
        draw_png_arrow(draw, pts, color, 4, dashed)

    draw_png_text(draw, 90, 1400, "Core figure recommendation: keep MMRotate outside this panel and present downstream detection as a separate application figure.", 22, PALETTE["muted"], False)
    return image


def render_detection_png() -> Image.Image:
    width, height = 2100, 980
    image = Image.new("RGB", (width, height), hex_rgb(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)

    draw_png_box(draw, BoxSpec(40, 28, 2020, 120, PALETTE["title"], PALETTE["title"], [], radius=28, stroke_width=1))
    draw_png_text(draw, 70, 40, "Downstream Rotated Object Detection Evaluation", 42, PALETTE["white"], True)
    draw_png_text(draw, 70, 94, "The detector platform is external to EMRDM and is used only for downstream evaluation", 22, "#d8e7f5", False)

    draw_png_box(draw, BoxSpec(40, 180, 2020, 740, PALETTE["panel"], PALETTE["navy"], [], radius=30, stroke_width=3))
    draw_png_text(draw, 70, 206, "Separate application figure", 32, PALETTE["navy"], True)

    for spec in [
        BoxSpec(90, 340, 260, 190, PALETTE["green_fill"], PALETTE["green"], ["Cloudy remote-sensing patch", "cloud / test image"], 28),
        BoxSpec(430, 300, 420, 270, PALETTE["blue_fill"], PALETTE["blue"], ["EMRDM restoration module", "Semantic-routed residual diffusion", "Produces restored cloud-free imagery"], 28),
        BoxSpec(930, 340, 280, 190, PALETTE["white"], PALETTE["blue"], ["Restored image", "for detector input"], 28),
        BoxSpec(1290, 260, 600, 360, "#fbfbfd", PALETTE["gray"], [], radius=28, stroke_width=3, dashed=True),
        BoxSpec(1350, 360, 480, 120, PALETTE["orange_fill"], PALETTE["orange"], ["Off-the-shelf rotated detector", "MMRotate / Faster R-CNN / YOLO-OBB"], 28),
        BoxSpec(1350, 520, 480, 70, "#fffaf5", PALETTE["orange"], ["Not a learnable component of EMRDM"], 24),
        BoxSpec(930, 680, 320, 120, PALETTE["purple_fill"], PALETTE["purple"], ["Detection outputs", "rotated boxes / category scores"], 26),
        BoxSpec(1350, 680, 480, 120, PALETTE["white"], PALETTE["purple"], ["Evaluation metrics", "mAP, AP50, downstream robustness"], 26),
        BoxSpec(90, 680, 760, 120, PALETTE["green_fill"], PALETTE["green"], ["Optional annotation conversion and dataset adapters", "for iSAID / DOTA / COCO / OBB evaluation pipelines"], 24),
        BoxSpec(470, 250, 340, 44, PALETTE["blue_fill"], PALETTE["blue"], ["Main paper method"], 20, radius=18, stroke_width=2),
        BoxSpec(1450, 250, 270, 44, "#f7f7f9", PALETTE["gray"], ["External evaluation tool"], 20, radius=18, stroke_width=2),
    ]:
        draw_png_box(draw, spec)

    draw_png_text(draw, 1320, 286, "External detector toolbox", 30, PALETTE["gray"], True)
    for pts, color, dashed in [
        ([(350, 435), (430, 435)], PALETTE["navy"], False),
        ([(850, 435), (930, 435)], PALETTE["navy"], False),
        ([(1210, 435), (1350, 435)], PALETTE["navy"], False),
        ([(1590, 480), (1590, 680), (1250, 680), (1250, 740)], PALETTE["orange"], False),
        ([(1590, 590), (1590, 680), (1830, 680), (1830, 740)], PALETTE["orange"], False),
        ([(850, 740), (930, 740)], PALETTE["green"], True),
    ]:
        draw_png_arrow(draw, pts, color, 5, dashed)

    draw_png_text(draw, 90, 875, "Recommended usage in the paper: keep this panel separate from the core EMRDM framework and describe the detector as an external downstream benchmark.", 22, PALETTE["muted"], False)
    return image


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    MAIN_SVG.write_text(render_main_svg(), encoding="utf-8")
    DET_SVG.write_text(render_detection_svg(), encoding="utf-8")
    render_main_png().save(MAIN_PNG)
    render_detection_png().save(DET_PNG)
    print(MAIN_SVG)
    print(MAIN_PNG)
    print(DET_SVG)
    print(DET_PNG)


if __name__ == "__main__":
    main()
