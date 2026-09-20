from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("D:/EMRDM-main/docs")
SVG_PATH = OUT_DIR / "emrdm_advanced_module_figure.svg"
PNG_PATH = OUT_DIR / "emrdm_advanced_module_figure.png"

W, H = 2400, 1400

COLORS = {
    "bg0": "#f7f2e9",
    "bg1": "#eef5fb",
    "ink": "#16324a",
    "muted": "#5f7082",
    "blue": "#24527a",
    "navy": "#24527a",
    "blue2": "#5f8fd3",
    "blue_fill": "#edf4ff",
    "green": "#3d8a69",
    "green_fill": "#eef8f2",
    "orange": "#c87634",
    "orange_fill": "#fff2e8",
    "purple": "#8f68bd",
    "purple_fill": "#f5efff",
    "gray": "#728091",
    "gray_fill": "#f3f5f7",
    "white": "#ffffff",
    "shadow": "#d6dee8",
    "cloud": "#ffffff",
    "grid": "#9db4c7",
    "highlight": "#f4c36a",
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False):
    candidates = (
        [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def svg_text(x: int, y: int, text: str, size: int, color: str, weight: str = "700", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-family="Arial, Segoe UI, sans-serif" font-weight="{weight}" text-anchor="{anchor}">{escape(text)}</text>'
    )


def svg_round_rect(x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: int = 2, r: int = 22, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def svg_polyline(points: Sequence[tuple[int, int]], color: str, width: int = 4, dashed: bool = False) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="11 8"' if dashed else ""
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"{dash}/>'
    )


def draw_text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: Sequence[str], size: int, fill: str, bold: bool = True, gap: int = 8):
    fnt = font(size, bold)
    x, y, w, h = box
    metrics = []
    total_h = 0
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=fnt)
        lh = bb[3] - bb[1]
        lw = bb[2] - bb[0]
        metrics.append((line, lw, lh))
        total_h += lh
    total_h += gap * (len(lines) - 1)
    cy = y + (h - total_h) / 2
    for line, lw, lh in metrics:
        cx = x + (w - lw) / 2
        draw.text((cx, cy), line, font=fnt, fill=rgb(fill))
        cy += lh + gap


def draw_round_rect(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: int = 2, r: int = 22):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=r, fill=rgb(fill), outline=rgb(stroke), width=sw)


def draw_arrow(draw: ImageDraw.ImageDraw, pts: Sequence[tuple[int, int]], color: str, width: int = 4, dashed: bool = False):
    c = rgb(color)
    if dashed:
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx, dy = x2 - x1, y2 - y1
            length = max((dx * dx + dy * dy) ** 0.5, 1.0)
            ux, uy = dx / length, dy / length
            pos = 0.0
            while pos < length:
                end = min(length, pos + 12)
                sx, sy = x1 + ux * pos, y1 + uy * pos
                ex, ey = x1 + ux * end, y1 + uy * end
                draw.line((sx, sy, ex, ey), fill=c, width=width)
                pos += 20
    else:
        draw.line(pts, fill=c, width=width)
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tri = [
        (x2, y2),
        (x2 - ux * 18 + px * 8, y2 - uy * 18 + py * 8),
        (x2 - ux * 18 - px * 8, y2 - uy * 18 - py * 8),
    ]
    draw.polygon(tri, fill=c)


def cloud_path_svg(cx: int, cy: int, scale: float = 1.0, fill: str = "#ffffff", stroke: str = "#b8c8d8") -> str:
    w = 120 * scale
    h = 58 * scale
    x = cx - w / 2
    y = cy - h / 2
    return (
        f'<path d="M {x+18},{y+h*0.75} '
        f'C {x+4},{y+h*0.75} {x},{y+h*0.58} {x},{y+h*0.48} '
        f'C {x},{y+h*0.34} {x+14},{y+h*0.25} {x+30},{y+h*0.25} '
        f'C {x+34},{y+h*0.07} {x+54},{y} {x+74},{y+h*0.08} '
        f'C {x+82},{y-h*0.02} {x+104},{y+h*0.1} {x+104},{y+h*0.28} '
        f'C {x+114},{y+h*0.3} {x+w},{y+h*0.42} {x+w},{y+h*0.58} '
        f'C {x+w},{y+h*0.7} {x+w-10},{y+h*0.78} {x+w-22},{y+h*0.78} Z" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )


def draw_cloud(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float = 1.0):
    w = int(120 * scale)
    h = int(58 * scale)
    x = cx - w // 2
    y = cy - h // 2
    fill = rgb(COLORS["cloud"])
    stroke = rgb("#b8c8d8")
    draw.ellipse((x + 10, y + 18, x + 48, y + 48), fill=fill, outline=stroke, width=2)
    draw.ellipse((x + 30, y + 6, x + 76, y + 44), fill=fill, outline=stroke, width=2)
    draw.ellipse((x + 62, y + 14, x + 106, y + 50), fill=fill, outline=stroke, width=2)
    draw.rounded_rectangle((x + 18, y + 26, x + 98, y + 52), radius=12, fill=fill, outline=stroke, width=2)


def svg_image_card(x: int, y: int, w: int, h: int, title: str, kind: str) -> str:
    parts = [svg_round_rect(x, y, w, h, COLORS["white"], COLORS["green"], 2, 22)]
    parts.append(svg_text(x + 18, y + 36, title, 24, COLORS["ink"], "700"))
    img_x, img_y, img_w, img_h = x + 18, y + 56, w - 36, h - 74
    parts.append(svg_round_rect(img_x, img_y, img_w, img_h, "#e5edf5", "#9cb3c6", 1, 18))
    if kind == "cloudy":
        parts.append(f'<defs><linearGradient id="g_cloudy" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#78a7cc"/><stop offset="100%" stop-color="#49759a"/></linearGradient></defs>')
        parts.append(svg_round_rect(img_x + 8, img_y + 8, img_w - 16, img_h - 16, "url(#g_cloudy)", "#6d8daa", 0, 14))
        for cx, cy, s in [(img_x + 78, img_y + 72, 0.9), (img_x + 165, img_y + 45, 0.7), (img_x + 140, img_y + 92, 1.0)]:
            parts.append(cloud_path_svg(cx, cy, s))
    elif kind == "semantic":
        colors = ["#4c8b74", "#7ab892", "#d7a55f", "#7b9ac8", "#e5d0ad"]
        cell = 28
        for r in range(ceil((img_h - 20) / cell)):
            for c in range(ceil((img_w - 20) / cell)):
                cx = img_x + 10 + c * cell
                cy = img_y + 10 + r * cell
                color = colors[(r * 2 + c) % len(colors)]
                parts.append(svg_round_rect(cx, cy, cell - 2, cell - 2, color, color, 0, 6))
    elif kind == "mask":
        parts.append(svg_round_rect(img_x + 8, img_y + 8, img_w - 16, img_h - 16, "#edf4ff", "#9cb3c6", 0, 14))
        for i in range(5):
            px = img_x + 24 + i * 34
            py = img_y + 22 + (i % 2) * 18
            parts.append(f'<circle cx="{px}" cy="{py}" r="18" fill="#d8e7f7"/>')
        parts.append(f'<path d="M {img_x+32},{img_y+86} C {img_x+64},{img_y+44} {img_x+116},{img_y+110} {img_x+168},{img_y+68} L {img_x+168},{img_y+132} L {img_x+32},{img_y+132} Z" fill="#4f8668" opacity="0.9"/>')
    return "\n".join(parts)


def draw_image_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, kind: str):
    draw_round_rect(draw, x, y, w, h, COLORS["white"], COLORS["green"], 2, 22)
    draw.text((x + 18, y + 14), title, font=font(24, True), fill=rgb(COLORS["ink"]))
    img_x, img_y, img_w, img_h = x + 18, y + 56, w - 36, h - 74
    draw_round_rect(draw, img_x, img_y, img_w, img_h, "#e5edf5", "#9cb3c6", 1, 18)
    if kind == "cloudy":
        for yy in range(img_y + 8, img_y + img_h - 8):
            t = (yy - (img_y + 8)) / max(img_h - 16, 1)
            c0 = rgb("#78a7cc")
            c1 = rgb("#49759a")
            c = tuple(int(c0[i] * (1 - t) + c1[i] * t) for i in range(3))
            draw.line((img_x + 8, yy, img_x + img_w - 8, yy), fill=c, width=1)
        for cx, cy, s in [(img_x + 78, img_y + 72, 0.9), (img_x + 165, img_y + 45, 0.7), (img_x + 140, img_y + 92, 1.0)]:
            draw_cloud(draw, cx, cy, s)
    elif kind == "semantic":
        colors = ["#4c8b74", "#7ab892", "#d7a55f", "#7b9ac8", "#e5d0ad"]
        cell = 28
        for r in range(ceil((img_h - 20) / cell)):
            for c in range(ceil((img_w - 20) / cell)):
                cx = img_x + 10 + c * cell
                cy = img_y + 10 + r * cell
                color = colors[(r * 2 + c) % len(colors)]
                draw_round_rect(draw, cx, cy, cell - 2, cell - 2, color, color, 0, 6)
    elif kind == "mask":
        draw_round_rect(draw, img_x + 8, img_y + 8, img_w - 16, img_h - 16, "#edf4ff", "#9cb3c6", 0, 14)
        for i in range(5):
            px = img_x + 24 + i * 34
            py = img_y + 22 + (i % 2) * 18
            draw.ellipse((px - 18, py - 18, px + 18, py + 18), fill=rgb("#d8e7f7"), outline=None)
        draw.polygon(
            [
                (img_x + 32, img_y + 86),
                (img_x + 64, img_y + 44),
                (img_x + 116, img_y + 110),
                (img_x + 168, img_y + 68),
                (img_x + 168, img_y + 132),
                (img_x + 32, img_y + 132),
            ],
            fill=rgb("#4f8668"),
        )


def svg_feature_panel(x: int, y: int, w: int, h: int) -> str:
    parts = [svg_round_rect(x, y, w, h, COLORS["blue_fill"], COLORS["blue"], 3, 30)]
    parts.append(svg_text(x + 28, y + 44, "Proposed restoration core", 34, COLORS["blue"], "700"))
    parts.append(svg_text(x + 28, y + 78, "Semantic-routed residual diffusion with dual-branch patch inference", 22, COLORS["muted"], "500"))

    # route fusion
    parts.append(svg_round_rect(x + 34, y + 120, 270, 180, COLORS["green_fill"], COLORS["green"], 2, 24))
    parts.append(svg_text(x + 58, y + 156, "Route fusion", 28, COLORS["green"], "700"))
    gx, gy = x + 72, y + 180
    for r in range(4):
        for c in range(4):
            fill = COLORS["highlight"] if (r + c) % 3 == 0 else "#d7e6da"
            parts.append(svg_round_rect(gx + c * 34, gy + r * 34, 28, 28, fill, "#9fb9aa", 1, 6))
    parts.append(svg_text(x + 198, y + 196, "semantic mask", 20, COLORS["ink"], "600"))
    parts.append(svg_text(x + 198, y + 224, "U  cloud mask", 20, COLORS["ink"], "600"))
    parts.append(svg_text(x + 198, y + 252, "route patch size = 128", 18, COLORS["muted"], "500"))

    # patch routing
    parts.append(svg_round_rect(x + 354, y + 120, 250, 180, COLORS["gray_fill"], COLORS["gray"], 2, 24))
    parts.append(svg_text(x + 382, y + 156, "Patch routing", 28, COLORS["gray"], "700"))
    rgx, rgy = x + 394, y + 182
    for r in range(3):
        for c in range(5):
            fill = COLORS["orange_fill"] if (r == 1 and c in (1, 3)) or (r == 0 and c == 4) else "#e7ecf2"
            stroke = COLORS["orange"] if fill == COLORS["orange_fill"] else "#b5c0cb"
            parts.append(svg_round_rect(rgx + c * 34, rgy + r * 34, 28, 28, fill, stroke, 1, 6))
    parts.append(svg_text(x + 378, y + 252, "route mode = max", 18, COLORS["ink"], "600"))
    parts.append(svg_text(x + 378, y + 278, "threshold = 0.005", 18, COLORS["ink"], "600"))

    # dual branch
    parts.append(svg_round_rect(x + 654, y + 110, 520, 210, COLORS["white"], COLORS["navy"], 2, 26))
    parts.append(svg_text(x + 680, y + 146, "Dual-branch denoiser", 30, COLORS["navy"], "700"))
    # light
    parts.append(svg_round_rect(x + 690, y + 184, 180, 102, COLORS["orange_fill"], COLORS["orange"], 2, 22))
    parts.append(svg_text(x + 750, y + 220, "Light branch", 24, COLORS["orange"], "700", "middle"))
    for i in range(3):
        parts.append(svg_round_rect(x + 724 + i * 36, y + 238, 22, 26, "#ffd9bf", COLORS["orange"], 1, 6))
    parts.append(svg_text(x + 780, y + 278, "background denoising", 18, COLORS["muted"], "500", "middle"))
    # heavy
    parts.append(svg_round_rect(x + 950, y + 174, 190, 120, COLORS["purple_fill"], COLORS["purple"], 2, 22))
    parts.append(svg_text(x + 1045, y + 214, "Heavy branch", 24, COLORS["purple"], "700", "middle"))
    for i, hh in enumerate([36, 48, 64, 78]):
        px = x + 982 + i * 32
        py = y + 260 - hh
        parts.append(svg_round_rect(px, py, 18, hh, "#dbcdf0", COLORS["purple"], 1, 6))
    parts.append(svg_text(x + 1045, y + 278, "ImageTransformer backbone", 18, COLORS["muted"], "500", "middle"))

    # merge
    parts.append(svg_round_rect(x + 640, y + 368, 560, 132, "#fbfcff", COLORS["navy"], 2, 24))
    parts.append(svg_text(x + 670, y + 408, "Patch merge and image reconstruction", 28, COLORS["navy"], "700"))
    mgx, mgy = x + 704, y + 430
    for r in range(3):
        for c in range(8):
            fill = "#dce8f7" if (r + c) % 4 else "#f4c36a"
            parts.append(svg_round_rect(mgx + c * 28, mgy + r * 20, 22, 14, fill, "#aabed4", 1, 4))
    parts.append(svg_text(x + 980, y + 454, "heavy patches replace selected regions", 18, COLORS["muted"], "500"))

    # arrows inside
    parts.append(svg_polyline([(x + 304, y + 210), (x + 354, y + 210)], COLORS["navy"], 4))
    parts.append(svg_polyline([(x + 604, y + 210), (x + 654, y + 210)], COLORS["navy"], 4))
    parts.append(svg_polyline([(x + 870, y + 234), (x + 950, y + 234)], COLORS["navy"], 4))
    parts.append(svg_polyline([(x + 780, y + 286), (x + 780, y + 368)], COLORS["navy"], 4))
    parts.append(svg_polyline([(x + 1045, y + 294), (x + 1045, y + 368)], COLORS["navy"], 4))
    return "\n".join(parts)


def draw_feature_panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int):
    draw_round_rect(draw, x, y, w, h, COLORS["blue_fill"], COLORS["blue"], 3, 30)
    draw.text((x + 28, y + 18), "Proposed restoration core", font=font(34, True), fill=rgb(COLORS["blue"]))
    draw.text((x + 28, y + 56), "Semantic-routed residual diffusion with dual-branch patch inference", font=font(22, False), fill=rgb(COLORS["muted"]))

    draw_round_rect(draw, x + 34, y + 120, 270, 180, COLORS["green_fill"], COLORS["green"], 2, 24)
    draw.text((x + 58, y + 142), "Route fusion", font=font(28, True), fill=rgb(COLORS["green"]))
    gx, gy = x + 72, y + 180
    for r in range(4):
        for c in range(4):
            fill = COLORS["highlight"] if (r + c) % 3 == 0 else "#d7e6da"
            draw_round_rect(draw, gx + c * 34, gy + r * 34, 28, 28, fill, "#9fb9aa", 1, 6)
    for i, txt in enumerate(["semantic mask", "U  cloud mask", "route patch size = 128"]):
        draw.text(
            (x + 198, y + 178 + i * 28),
            txt,
            font=font(18 if i == 2 else 20, i != 2),
            fill=rgb(COLORS["ink"] if i < 2 else COLORS["muted"]),
        )

    draw_round_rect(draw, x + 354, y + 120, 250, 180, COLORS["gray_fill"], COLORS["gray"], 2, 24)
    draw.text((x + 382, y + 142), "Patch routing", font=font(28, True), fill=rgb(COLORS["gray"]))
    rgx, rgy = x + 394, y + 182
    for r in range(3):
        for c in range(5):
            fill = COLORS["orange_fill"] if (r == 1 and c in (1, 3)) or (r == 0 and c == 4) else "#e7ecf2"
            stroke = COLORS["orange"] if fill == COLORS["orange_fill"] else "#b5c0cb"
            draw_round_rect(draw, rgx + c * 34, rgy + r * 34, 28, 28, fill, stroke, 1, 6)
    draw.text((x + 378, y + 248), "route mode = max", font=font(18, True), fill=rgb(COLORS["ink"]))
    draw.text((x + 378, y + 274), "threshold = 0.005", font=font(18, True), fill=rgb(COLORS["ink"]))

    draw_round_rect(draw, x + 654, y + 110, 520, 210, COLORS["white"], COLORS["navy"], 2, 26)
    draw.text((x + 680, y + 130), "Dual-branch denoiser", font=font(30, True), fill=rgb(COLORS["navy"]))
    draw_round_rect(draw, x + 690, y + 184, 180, 102, COLORS["orange_fill"], COLORS["orange"], 2, 22)
    draw_text_center(draw, (x + 690, y + 184, 180, 58), ["Light branch"], 24, COLORS["orange"], True)
    for i in range(3):
        draw_round_rect(draw, x + 724 + i * 36, y + 238, 22, 26, "#ffd9bf", COLORS["orange"], 1, 6)
    draw_text_center(draw, (x + 700, y + 264, 160, 24), ["background denoising"], 18, COLORS["muted"], False, 0)
    draw_round_rect(draw, x + 950, y + 174, 190, 120, COLORS["purple_fill"], COLORS["purple"], 2, 22)
    draw_text_center(draw, (x + 950, y + 180, 190, 44), ["Heavy branch"], 24, COLORS["purple"], True, 0)
    for i, hh in enumerate([36, 48, 64, 78]):
        px = x + 982 + i * 32
        py = y + 260 - hh
        draw_round_rect(draw, px, py, 18, hh, "#dbcdf0", COLORS["purple"], 1, 6)
    draw_text_center(draw, (x + 962, y + 264, 166, 24), ["ImageTransformer backbone"], 18, COLORS["muted"], False, 0)

    draw_round_rect(draw, x + 640, y + 368, 560, 132, "#fbfcff", COLORS["navy"], 2, 24)
    draw.text((x + 670, y + 392), "Patch merge and image reconstruction", font=font(28, True), fill=rgb(COLORS["navy"]))
    mgx, mgy = x + 704, y + 430
    for r in range(3):
        for c in range(8):
            fill = "#dce8f7" if (r + c) % 4 else "#f4c36a"
            draw_round_rect(draw, mgx + c * 28, mgy + r * 20, 22, 14, fill, "#aabed4", 1, 4)
    draw.text((x + 980, y + 448), "heavy patches replace selected regions", font=font(18, False), fill=rgb(COLORS["muted"]))

    for pts in [
        [(x + 304, y + 210), (x + 354, y + 210)],
        [(x + 604, y + 210), (x + 654, y + 210)],
        [(x + 870, y + 234), (x + 950, y + 234)],
        [(x + 780, y + 286), (x + 780, y + 368)],
        [(x + 1045, y + 294), (x + 1045, y + 368)],
    ]:
        draw_arrow(draw, pts, COLORS["navy"], 4)


def svg_loss_panel(x: int, y: int, w: int, h: int) -> str:
    parts = [svg_round_rect(x, y, w, h, COLORS["orange_fill"], COLORS["orange"], 3, 28)]
    parts.append(svg_text(x + 28, y + 44, "Innovation-aware objectives", 32, COLORS["orange"], "700"))
    parts.append(svg_text(x + 28, y + 78, "Loss branches are shown as modules rather than plain text descriptions", 20, COLORS["muted"], "500"))

    cards = [
        (x + 34, y + 120, 330, 150, COLORS["white"], COLORS["orange"], ["Cloud-aware", "importance loss"], "cloud"),
        (x + 34, y + 304, 330, 170, COLORS["white"], COLORS["purple"], ["Asymmetric semantic", "feature matching"], "feature"),
        (x + 34, y + 508, 330, 100, "#fffaf5", COLORS["orange"], ["Residual EDM weighting"], "weight"),
        (x + 34, y + 640, 330, 92, "#fffdf8", COLORS["navy"], ["Total restoration objective"], "sum"),
    ]
    for cx, cy, cw, ch, fill, stroke, lines, icon in cards:
        parts.append(svg_round_rect(cx, cy, cw, ch, fill, stroke, 2, 22))
        parts.append(svg_text(cx + 28, cy + 42, lines[0], 26, stroke, "700"))
        if len(lines) > 1:
            parts.append(svg_text(cx + 28, cy + 74, lines[1], 26, stroke, "700"))
        if icon == "cloud":
            parts.append(cloud_path_svg(cx + 262, cy + 88, 0.8, "#ffffff", "#b8c8d8"))
            parts.append(svg_text(cx + 28, cy + 108, "higher weights on cloud-covered regions", 18, COLORS["muted"], "500"))
        elif icon == "feature":
            for i in range(4):
                parts.append(svg_round_rect(cx + 210 + i * 22, cy + 50 + i * 10, 18, 64 - i * 10, "#d9ccf0", COLORS["purple"], 1, 5))
            parts.append(svg_text(cx + 28, cy + 112, "semantic foreground and background", 18, COLORS["muted"], "500"))
            parts.append(svg_text(cx + 28, cy + 138, "are weighted differently in VGG space", 18, COLORS["muted"], "500"))
        elif icon == "weight":
            for i, r in enumerate([28, 22, 16]):
                parts.append(f'<circle cx="{cx+270}" cy="{cy+54}" r="{r}" fill="none" stroke="{COLORS["orange"]}" stroke-width="2" opacity="{1 - i * 0.18}"/>')
            parts.append(svg_text(cx + 28, cy + 74, "sigma-aware residual weighting", 18, COLORS["muted"], "500"))
        else:
            parts.append(svg_text(cx + 28, cy + 72, "pixel reconstruction + 0.10 * semantic FM", 18, COLORS["muted"], "500"))

    parts.append(svg_polyline([(x + 200, y + 270), (x + 200, y + 304)], COLORS["orange"], 4))
    parts.append(svg_polyline([(x + 200, y + 474), (x + 200, y + 508)], COLORS["orange"], 4))
    parts.append(svg_polyline([(x + 200, y + 608), (x + 200, y + 640)], COLORS["orange"], 4))
    return "\n".join(parts)


def draw_loss_panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int):
    draw_round_rect(draw, x, y, w, h, COLORS["orange_fill"], COLORS["orange"], 3, 28)
    draw.text((x + 28, y + 18), "Innovation-aware objectives", font=font(32, True), fill=rgb(COLORS["orange"]))
    draw.text((x + 28, y + 56), "Loss branches are shown as modules rather than plain text descriptions", font=font(20, False), fill=rgb(COLORS["muted"]))
    cards = [
        (x + 34, y + 120, 330, 150, COLORS["white"], COLORS["orange"], ["Cloud-aware", "importance loss"], "cloud"),
        (x + 34, y + 304, 330, 170, COLORS["white"], COLORS["purple"], ["Asymmetric semantic", "feature matching"], "feature"),
        (x + 34, y + 508, 330, 100, "#fffaf5", COLORS["orange"], ["Residual EDM weighting"], "weight"),
        (x + 34, y + 640, 330, 92, "#fffdf8", COLORS["navy"], ["Total restoration objective"], "sum"),
    ]
    for cx, cy, cw, ch, fill, stroke, lines, icon in cards:
        draw_round_rect(draw, cx, cy, cw, ch, fill, stroke, 2, 22)
        draw.text((cx + 28, cy + 22), lines[0], font=font(26, True), fill=rgb(stroke))
        if len(lines) > 1:
            draw.text((cx + 28, cy + 54), lines[1], font=font(26, True), fill=rgb(stroke))
        if icon == "cloud":
            draw_cloud(draw, cx + 262, cy + 88, 0.8)
            draw.text((cx + 28, cy + 98), "higher weights on cloud-covered regions", font=font(18, False), fill=rgb(COLORS["muted"]))
        elif icon == "feature":
            for i in range(4):
                draw_round_rect(draw, cx + 210 + i * 22, cy + 50 + i * 10, 18, 64 - i * 10, "#d9ccf0", COLORS["purple"], 1, 5)
            draw.text((cx + 28, cy + 108), "semantic foreground and background", font=font(18, False), fill=rgb(COLORS["muted"]))
            draw.text((cx + 28, cy + 132), "are weighted differently in VGG space", font=font(18, False), fill=rgb(COLORS["muted"]))
        elif icon == "weight":
            for i, r in enumerate([28, 22, 16]):
                bbox = (cx + 270 - r, cy + 54 - r, cx + 270 + r, cy + 54 + r)
                draw.ellipse(bbox, outline=rgb(COLORS["orange"]), width=2)
            draw.text((cx + 28, cy + 66), "sigma-aware residual weighting", font=font(18, False), fill=rgb(COLORS["muted"]))
        else:
            draw.text((cx + 28, cy + 60), "pixel reconstruction + 0.10 * semantic FM", font=font(18, False), fill=rgb(COLORS["muted"]))
    for pts in [
        [(x + 200, y + 270), (x + 200, y + 304)],
        [(x + 200, y + 474), (x + 200, y + 508)],
        [(x + 200, y + 608), (x + 200, y + 640)],
    ]:
        draw_arrow(draw, pts, COLORS["orange"], 4)


def svg_infer_panel(x: int, y: int, w: int, h: int) -> str:
    parts = [svg_round_rect(x, y, w, h, COLORS["green_fill"], COLORS["green"], 3, 28)]
    parts.append(svg_text(x + 28, y + 44, "Inference and restored output", 30, COLORS["green"], "700"))
    parts.append(svg_text(x + 28, y + 76, "The same semantic-routed module is reused during reverse diffusion", 20, COLORS["muted"], "500"))

    # sampler icon
    parts.append(svg_round_rect(x + 42, y + 118, 240, 110, COLORS["white"], COLORS["green"], 2, 20))
    parts.append(svg_text(x + 70, y + 156, "Residual Heun", 26, COLORS["green"], "700"))
    parts.append(svg_text(x + 70, y + 188, "EDM sampler", 26, COLORS["green"], "700"))
    for i in range(8):
        cx = x + 72 + i * 18
        parts.append(f'<circle cx="{cx}" cy="{y+206}" r="5" fill="{COLORS["green"]}" opacity="{0.4 + i * 0.07}"/>')

    parts.append(svg_round_rect(x + 340, y + 118, 330, 110, COLORS["white"], COLORS["green"], 2, 20))
    parts.append(svg_text(x + 368, y + 156, "Semantic-routed denoiser", 24, COLORS["navy"], "700"))
    parts.append(svg_text(x + 368, y + 188, "semantic_mask + cloud mask stay active", 18, COLORS["muted"], "500"))

    parts.append(svg_round_rect(x + 730, y + 102, 250, 126, COLORS["white"], COLORS["green"], 2, 20))
    parts.append(svg_text(x + 760, y + 146, "Restored image", 28, COLORS["green"], "700"))
    parts.append(svg_round_rect(x + 780, y + 164, 150, 42, "#dce8f7", "#adc0d3", 1, 12))
    parts.append(svg_text(x + 855, y + 193, "cloud-free output", 18, COLORS["ink"], "600", "middle"))

    # small image-like output
    parts.append(svg_round_rect(x + 1040, y + 84, 300, 162, COLORS["white"], COLORS["green"], 2, 22))
    parts.append(svg_text(x + 1068, y + 120, "Visual output and downstream use", 24, COLORS["green"], "700"))
    parts.append(svg_round_rect(x + 1070, y + 140, 110, 80, "#6f9fc5", "#8ba6bc", 0, 14))
    parts.append(svg_round_rect(x + 1200, y + 140, 110, 80, "#e5edf5", "#8ba6bc", 1, 14))
    parts.append(cloud_path_svg(x + 1124, y + 184, 0.55))
    for i in range(3):
        parts.append(svg_round_rect(x + 1220 + i * 24, y + 164 - i * 8, 16, 42 + i * 8, "#d9ccf0", COLORS["purple"], 1, 4))

    parts.append(svg_polyline([(x + 282, y + 172), (x + 340, y + 172)], COLORS["green"], 4))
    parts.append(svg_polyline([(x + 670, y + 172), (x + 730, y + 172)], COLORS["green"], 4))
    parts.append(svg_polyline([(x + 980, y + 172), (x + 1040, y + 172)], COLORS["green"], 4))
    return "\n".join(parts)


def draw_infer_panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int):
    draw_round_rect(draw, x, y, w, h, COLORS["green_fill"], COLORS["green"], 3, 28)
    draw.text((x + 28, y + 18), "Inference and restored output", font=font(30, True), fill=rgb(COLORS["green"]))
    draw.text((x + 28, y + 52), "The same semantic-routed module is reused during reverse diffusion", font=font(20, False), fill=rgb(COLORS["muted"]))

    draw_round_rect(draw, x + 42, y + 118, 240, 110, COLORS["white"], COLORS["green"], 2, 20)
    draw.text((x + 70, y + 136), "Residual Heun", font=font(26, True), fill=rgb(COLORS["green"]))
    draw.text((x + 70, y + 168), "EDM sampler", font=font(26, True), fill=rgb(COLORS["green"]))
    for i in range(8):
        cx = x + 72 + i * 18
        draw.ellipse((cx - 5, y + 201 - 5, cx + 5, y + 201 + 5), fill=rgb(COLORS["green"]))

    draw_round_rect(draw, x + 340, y + 118, 330, 110, COLORS["white"], COLORS["green"], 2, 20)
    draw.text((x + 368, y + 136), "Semantic-routed denoiser", font=font(24, True), fill=rgb(COLORS["navy"]))
    draw.text((x + 368, y + 170), "semantic_mask + cloud mask stay active", font=font(18, False), fill=rgb(COLORS["muted"]))

    draw_round_rect(draw, x + 730, y + 102, 250, 126, COLORS["white"], COLORS["green"], 2, 20)
    draw.text((x + 760, y + 136), "Restored image", font=font(28, True), fill=rgb(COLORS["green"]))
    draw_round_rect(draw, x + 780, y + 164, 150, 42, "#dce8f7", "#adc0d3", 1, 12)
    draw_text_center(draw, (x + 780, y + 164, 150, 42), ["cloud-free output"], 18, COLORS["ink"], True, 0)

    draw_round_rect(draw, x + 1040, y + 84, 300, 162, COLORS["white"], COLORS["green"], 2, 22)
    draw.text((x + 1068, y + 102), "Visual output and downstream use", font=font(24, True), fill=rgb(COLORS["green"]))
    draw_round_rect(draw, x + 1070, y + 140, 110, 80, "#6f9fc5", "#8ba6bc", 0, 14)
    draw_round_rect(draw, x + 1200, y + 140, 110, 80, "#e5edf5", "#8ba6bc", 1, 14)
    draw_cloud(draw, x + 1124, y + 184, 0.55)
    for i in range(3):
        draw_round_rect(draw, x + 1220 + i * 24, y + 164 - i * 8, 16, 42 + i * 8, "#d9ccf0", COLORS["purple"], 1, 4)

    for pts in [
        [(x + 282, y + 172), (x + 340, y + 172)],
        [(x + 670, y + 172), (x + 730, y + 172)],
        [(x + 980, y + 172), (x + 1040, y + 172)],
    ]:
        draw_arrow(draw, pts, COLORS["green"], 4)


def build_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#f7f2e9"/><stop offset="52%" stop-color="#eef5fb"/><stop offset="100%" stop-color="#f8fbf7"/></linearGradient>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#24527a"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
        svg_round_rect(36, 24, 2328, 118, COLORS["blue"], COLORS["blue"], 1, 30),
        svg_text(62, 78, "EMRDM Advanced Module Figure", 46, COLORS["white"], "700"),
        svg_text(62, 118, "Semantic-routed residual diffusion with explicit innovation modules", 22, "#d9e7f4", "500"),
        svg_round_rect(36, 174, 2328, 1188, COLORS["white"], COLORS["navy"], 3, 34),
        svg_text(62, 232, "Publication-style architecture with visual modules", 34, COLORS["navy"], "700"),
        svg_text(62, 268, "Only the proposed restoration framework is shown; external detectors should be presented separately.", 20, COLORS["muted"], "500"),
    ]

    parts.append(svg_image_card(70, 330, 250, 230, "Cloudy input", "cloudy"))
    parts.append(svg_image_card(70, 596, 250, 230, "Semantic prior", "semantic"))
    parts.append(svg_image_card(70, 862, 250, 230, "Cloud prior", "mask"))
    parts.append(svg_polyline([(320, 446), (390, 446), (390, 520)], COLORS["green"], 4))
    parts.append(svg_polyline([(320, 712), (390, 712), (390, 560)], COLORS["green"], 4))
    parts.append(svg_polyline([(320, 978), (390, 978), (390, 600)], COLORS["green"], 4))

    parts.append(svg_feature_panel(360, 320, 1270, 760))
    parts.append(svg_loss_panel(1670, 320, 420, 760))
    parts.append(svg_polyline([(1630, 700), (1670, 700)], COLORS["navy"], 5))
    parts.append(svg_infer_panel(360, 1110, 1730, 210))

    # innovation badges
    badges = [
        (470, 286, 210, 46, COLORS["green_fill"], COLORS["green"], "I1  Route-mask fusion"),
        (702, 286, 230, 46, COLORS["blue_fill"], COLORS["blue"], "I2  Patch-level semantic routing"),
        (956, 286, 220, 46, COLORS["orange_fill"], COLORS["orange"], "I3  Dual-branch denoiser"),
        (1198, 286, 248, 46, COLORS["purple_fill"], COLORS["purple"], "I4  Semantic-aware objective"),
    ]
    for x, y, w, h, fill, stroke, text in badges:
        parts.append(svg_round_rect(x, y, w, h, fill, stroke, 2, 18))
        parts.append(svg_text(x + w // 2, y + 31, text, 20, COLORS["ink"], "700", "middle"))

    parts.append(svg_text(72, 1336, "Tip: if you still want to show detection, place it in a second panel named 'Downstream rotated object detection evaluation' and mark it as external.", 20, COLORS["muted"], "500"))
    parts.append("</svg>")
    return "\n".join(parts)


def build_png() -> Image.Image:
    image = Image.new("RGB", (W, H), rgb(COLORS["bg0"]))
    draw = ImageDraw.Draw(image)

    for yy in range(H):
        t = yy / max(H - 1, 1)
        c0 = rgb(COLORS["bg0"])
        c1 = rgb(COLORS["bg1"])
        c = tuple(int(c0[i] * (1 - t) + c1[i] * t) for i in range(3))
        draw.line((0, yy, W, yy), fill=c)

    draw_round_rect(draw, 36, 24, 2328, 118, COLORS["blue"], COLORS["blue"], 1, 30)
    draw.text((62, 38), "EMRDM Advanced Module Figure", font=font(46, True), fill=rgb(COLORS["white"]))
    draw.text((62, 92), "Semantic-routed residual diffusion with explicit innovation modules", font=font(22, False), fill=rgb("#d9e7f4"))
    draw_round_rect(draw, 36, 174, 2328, 1188, COLORS["white"], COLORS["navy"], 3, 34)
    draw.text((62, 194), "Publication-style architecture with visual modules", font=font(34, True), fill=rgb(COLORS["navy"]))
    draw.text((62, 238), "Only the proposed restoration framework is shown; external detectors should be presented separately.", font=font(20, False), fill=rgb(COLORS["muted"]))

    draw_image_card(draw, 70, 330, 250, 230, "Cloudy input", "cloudy")
    draw_image_card(draw, 70, 596, 250, 230, "Semantic prior", "semantic")
    draw_image_card(draw, 70, 862, 250, 230, "Cloud prior", "mask")
    for pts in [
        [(320, 446), (390, 446), (390, 520)],
        [(320, 712), (390, 712), (390, 560)],
        [(320, 978), (390, 978), (390, 600)],
    ]:
        draw_arrow(draw, pts, COLORS["green"], 4)

    draw_feature_panel(draw, 360, 320, 1270, 760)
    draw_loss_panel(draw, 1670, 320, 420, 760)
    draw_arrow(draw, [(1630, 700), (1670, 700)], COLORS["navy"], 5)
    draw_infer_panel(draw, 360, 1110, 1730, 210)

    badges = [
        (470, 286, 210, 46, COLORS["green_fill"], COLORS["green"], "I1  Route-mask fusion"),
        (702, 286, 230, 46, COLORS["blue_fill"], COLORS["blue"], "I2  Patch-level semantic routing"),
        (956, 286, 220, 46, COLORS["orange_fill"], COLORS["orange"], "I3  Dual-branch denoiser"),
        (1198, 286, 248, 46, COLORS["purple_fill"], COLORS["purple"], "I4  Semantic-aware objective"),
    ]
    for x, y, w, h, fill, stroke, text in badges:
        draw_round_rect(draw, x, y, w, h, fill, stroke, 2, 18)
        draw_text_center(draw, (x, y, w, h), [text], 20, COLORS["ink"], True, 0)

    draw.text((72, 1336), "Tip: if you still want to show detection, place it in a second panel named 'Downstream rotated object detection evaluation' and mark it as external.", font=font(20, False), fill=rgb(COLORS["muted"]))
    return image


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    build_png().save(PNG_PATH)
    print(SVG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
