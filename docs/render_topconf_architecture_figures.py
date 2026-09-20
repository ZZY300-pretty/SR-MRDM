from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


PALETTE = {
    "bg": "#f6f3ec",
    "panel": "#fffdfa",
    "ink": "#1f2933",
    "muted": "#5b6874",
    "blue_fill": "#edf3ff",
    "blue_stroke": "#4b6fb6",
    "green_fill": "#eef8f1",
    "green_stroke": "#4d8a65",
    "orange_fill": "#fff3eb",
    "orange_stroke": "#c66b38",
    "purple_fill": "#f5efff",
    "purple_stroke": "#8a63b6",
    "gray_fill": "#f4f6f8",
    "gray_stroke": "#707b86",
    "header_fill": "#173b5c",
    "header_text": "#ffffff",
    "accent": "#224d73",
    "baseline": "#7a8794",
    "innovation": "#b3532f",
}


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    lines: Sequence[str]
    fill: str
    stroke: str
    title_size: int = 26
    body_size: int = 20
    radius: int = 22

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


class FigureBuilder:
    def __init__(self, width: int, height: int, title: str, subtitle: str):
        self.width = width
        self.height = height
        self.title = title
        self.subtitle = subtitle
        self.image = Image.new("RGB", (width, height), hex_to_rgb(PALETTE["bg"]))
        self.draw = ImageDraw.Draw(self.image)
        self.svg: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
            f'    <path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["accent"]}"/>',
            "  </marker>",
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>',
        ]
        self.font_title = load_font(42, bold=True)
        self.font_subtitle = load_font(22, bold=False)
        self.font_panel = load_font(32, bold=True)
        self.font_tag = load_font(18, bold=True)
        self._draw_header()

    def _draw_header(self) -> None:
        x, y, w, h = 40, 28, self.width - 80, 122
        self.rounded_rect(x, y, w, h, 30, PALETTE["header_fill"], PALETTE["header_fill"], 1)
        self.draw.text((70, 70), self.title, font=self.font_title, fill=hex_to_rgb(PALETTE["header_text"]))
        self.draw.text((70, 113), self.subtitle, font=self.font_subtitle, fill=(216, 231, 245))
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="30" ry="30" fill="{PALETTE["header_fill"]}" stroke="{PALETTE["header_fill"]}" stroke-width="1"/>'
        )
        self.svg.append(
            f'<text x="70" y="86" font-size="42" font-family="Arial, Segoe UI, sans-serif" fill="{PALETTE["header_text"]}" font-weight="700">{escape(self.title)}</text>'
        )
        self.svg.append(
            f'<text x="70" y="122" font-size="22" font-family="Arial, Segoe UI, sans-serif" fill="#d8e7f5">{escape(self.subtitle)}</text>'
        )

    def save(self, stem: str) -> None:
        png_path = ROOT / f"{stem}.png"
        svg_path = ROOT / f"{stem}.svg"
        self.image.save(png_path)
        self.svg.append("</svg>")
        svg_path.write_text("\n".join(self.svg), encoding="utf-8")

    def rounded_rect(
        self, x: int, y: int, w: int, h: int, r: int, fill: str, stroke: str, stroke_width: int
    ) -> None:
        self.draw.rounded_rectangle((x, y, x + w, y + h), radius=r, fill=hex_to_rgb(fill), outline=hex_to_rgb(stroke), width=stroke_width)
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def panel(self, x: int, y: int, w: int, h: int, title: str, fill: str, stroke: str) -> None:
        self.rounded_rect(x, y, w, h, 30, fill, stroke, 3)
        self.draw.text((x + 26, y + 26), title, font=self.font_panel, fill=hex_to_rgb(stroke))
        self.svg.append(
            f'<text x="{x+26}" y="{y+58}" font-size="32" font-family="Arial, Segoe UI, sans-serif" fill="{stroke}" font-weight="700">{escape(title)}</text>'
        )

    def tag(self, x: int, y: int, text: str, fill: str, stroke: str) -> None:
        pad_x = 14
        bbox = self.draw.textbbox((0, 0), text, font=self.font_tag)
        w = bbox[2] - bbox[0] + pad_x * 2
        h = 34
        self.rounded_rect(x, y, w, h, 16, fill, stroke, 2)
        self.draw.text((x + pad_x, y + 7), text, font=self.font_tag, fill=hex_to_rgb(PALETTE["ink"]))
        self.svg.append(
            f'<text x="{x+pad_x}" y="{y+23}" font-size="18" font-family="Arial, Segoe UI, sans-serif" fill="{PALETTE["ink"]}" font-weight="700">{escape(text)}</text>'
        )

    def box(self, box: Box) -> None:
        self.rounded_rect(box.x, box.y, box.w, box.h, box.radius, box.fill, box.stroke, 2)
        title_font = load_font(box.title_size, bold=True)
        body_font = load_font(box.body_size, bold=False)
        self.draw.text((box.x + box.w / 2, box.y + 18), box.title, anchor="ma", font=title_font, fill=hex_to_rgb(PALETTE["ink"]))
        self.svg.append(
            f'<text x="{box.cx}" y="{box.y+18+box.title_size}" text-anchor="middle" font-size="{box.title_size}" font-family="Arial, Segoe UI, sans-serif" fill="{PALETTE["ink"]}" font-weight="700">{escape(box.title)}</text>'
        )
        start_y = box.y + 56
        line_gap = box.body_size + 10
        for idx, line in enumerate(box.lines):
            yy = start_y + idx * line_gap
            self.draw.text((box.x + box.w / 2, yy), line, anchor="ma", font=body_font, fill=hex_to_rgb(PALETTE["ink"]))
            self.svg.append(
                f'<text x="{box.cx}" y="{yy+box.body_size}" text-anchor="middle" font-size="{box.body_size}" font-family="Arial, Segoe UI, sans-serif" fill="{PALETTE["ink"]}">{escape(line)}</text>'
            )

    def arrow(self, points: Sequence[Tuple[int, int]], stroke: Optional[str] = None, width: int = 4) -> None:
        stroke = stroke or PALETTE["accent"]
        self.draw.line(points, fill=hex_to_rgb(stroke), width=width, joint="curve")
        if len(points) >= 2:
            x1, y1 = points[-2]
            x2, y2 = points[-1]
            self._arrow_head(x1, y1, x2, y2, stroke)
        pts = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"/>'
        )

    def _arrow_head(self, x1: int, y1: int, x2: int, y2: int, stroke: str) -> None:
        # Rely on SVG marker; raster head is minimal and cosmetic only.
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return
        import math

        ang = math.atan2(dy, dx)
        size = 10
        a1 = ang + 2.7
        a2 = ang - 2.7
        p1 = (x2 + size * math.cos(a1), y2 + size * math.sin(a1))
        p2 = (x2 + size * math.cos(a2), y2 + size * math.sin(a2))
        self.draw.polygon([(x2, y2), p1, p2], fill=hex_to_rgb(stroke))


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def add_footer(fig: FigureBuilder, text: str) -> None:
    font = load_font(18, bold=False)
    fig.draw.text((fig.width - 50, fig.height - 34), text, anchor="ra", font=font, fill=hex_to_rgb(PALETTE["muted"]))
    fig.svg.append(
        f'<text x="{fig.width-50}" y="{fig.height-24}" text-anchor="end" font-size="18" font-family="Arial, Segoe UI, sans-serif" fill="{PALETTE["muted"]}">{escape(text)}</text>'
    )


def build_overview() -> None:
    fig = FigureBuilder(
        2400,
        1540,
        "Architecture of the Proposed Semantic-Routed EMRDM for Cloud Removal",
        "Baseline EMRDM is extended with Perlin-cloud synthesis, semantic routing, shared background prediction, overlap fusion, and asymmetric semantic supervision",
    )
    fig.panel(40, 180, 2320, 1320, "Overall Training and Inference Pipeline", PALETTE["panel"], PALETTE["accent"])
    fig.tag(122, 246, "B1 Baseline mean-reverting residual diffusion", PALETTE["gray_fill"], PALETTE["baseline"])
    fig.tag(500, 246, "I1 Perlin cloud synthesis", PALETTE["green_fill"], PALETTE["green_stroke"])
    fig.tag(760, 246, "I2 Semantic-routed dual branch", PALETTE["blue_fill"], PALETTE["blue_stroke"])
    fig.tag(1085, 246, "I3 Shared background + stride=64 overlap", PALETTE["orange_fill"], PALETTE["orange_stroke"])
    fig.tag(1530, 246, "I4 Asymmetric semantic feature matching", PALETTE["purple_fill"], PALETTE["purple_stroke"])
    fig.tag(1960, 246, "I5 Border exclusion", PALETTE["green_fill"], PALETTE["green_stroke"])

    fig.panel(90, 330, 360, 970, "Data Preparation", PALETTE["green_fill"], PALETTE["green_stroke"])
    boxes = [
        Box(125, 410, 290, 110, "Clean crop", ["Remote-sensing image patch", "label / x0"], "#ffffff", PALETTE["green_stroke"], body_size=18),
        Box(125, 555, 290, 136, "Perlin-fBm cloud synthesis", ["coarse + detail fBm", "warp_x / warp_y", "alpha mask and thin-cloud image"], "#ffffff", PALETTE["green_stroke"], body_size=17),
        Box(125, 725, 290, 110, "Auxiliary masks", ["cloud mask M", "semantic_mask"], "#ffffff", PALETTE["green_stroke"], body_size=18),
        Box(125, 870, 290, 128, "Border handling", ["crop padding", "1 = real image", "0 = padded border"], "#ffffff", PALETTE["green_stroke"], body_size=17),
        Box(125, 1035, 290, 170, "Training tuple", ["cond_image (cloudy)", "label (clear)", "M, semantic_mask"], "#f8fff9", PALETTE["green_stroke"], body_size=19),
    ]
    for b in boxes:
        fig.box(b)
    fig.arrow([(270, 520), (270, 555)])
    fig.arrow([(270, 691), (270, 725)])
    fig.arrow([(270, 835), (270, 870)])
    fig.arrow([(270, 998), (270, 1035)])

    fig.panel(500, 330, 1140, 570, "Residual Diffusion Restoration Core", PALETTE["blue_fill"], PALETTE["blue_stroke"])
    core_boxes = [
        Box(540, 415, 250, 116, "EMRDM baseline", ["x0 = label", "mu = cond_image"], "#ffffff", PALETTE["blue_stroke"], body_size=19),
        Box(830, 415, 240, 116, "Sigma-to-state", ["EDMSigma2St", "alpha = 3.0"], "#ffffff", PALETTE["blue_stroke"], body_size=19),
        Box(1110, 415, 250, 116, "Conditioner", ["Identity embedder", "for cond_image"], "#ffffff", PALETTE["blue_stroke"], body_size=19),
        Box(1400, 415, 200, 116, "Wrapper", ["concat(x_t, mu)", "6-channel input"], "#ffffff", PALETTE["blue_stroke"], body_size=19),
        Box(660, 600, 820, 220, "Semantic-Routed Patch Denoiser", ["Route mask = semantic_mask union M", "Light branch predicts whole-image background", "Heavy branch refines routed patches only", "Shared background prediction + overlap fusion"], "#ffffff", PALETTE["accent"], title_size=28, body_size=21),
    ]
    for b in core_boxes:
        fig.box(b)
    fig.arrow([(790, 473), (830, 473)])
    fig.arrow([(1070, 473), (1110, 473)])
    fig.arrow([(1360, 473), (1400, 473)])
    fig.arrow([(1500, 531), (1500, 565), (1070, 565), (1070, 600)])
    fig.arrow([(450, 1120), (520, 1120), (520, 473), (540, 473)])

    fig.panel(1680, 330, 630, 570, "Objective Design", PALETTE["orange_fill"], PALETTE["orange_stroke"])
    loss_boxes = [
        Box(1730, 415, 530, 126, "Cloud-aware reconstruction", ["importance-weighted pixel loss", "foreground cloud regions receive larger weights"], "#ffffff", PALETTE["orange_stroke"], body_size=18),
        Box(1730, 575, 530, 150, "Asymmetric semantic FM", ["VGG relu2_2 / relu3_3 / relu4_3", "foreground > background", "extra boost on cloud-covered semantic regions"], "#ffffff", PALETTE["orange_stroke"], body_size=18),
        Box(1730, 760, 530, 95, "Masked optimization", ["padded borders are ignored in training"], "#fffaf5", PALETTE["orange_stroke"], body_size=18),
    ]
    for b in loss_boxes:
        fig.box(b)
    fig.arrow([(1480, 710), (1620, 710), (1620, 478), (1730, 478)])
    fig.arrow([(1480, 710), (1620, 710), (1620, 650), (1730, 650)])
    fig.arrow([(1480, 710), (1620, 710), (1620, 807), (1730, 807)])

    fig.panel(500, 950, 1810, 350, "Inference Path", PALETTE["purple_fill"], PALETTE["purple_stroke"])
    infer_boxes = [
        Box(560, 1038, 300, 130, "Residual Heun sampler", ["8 reverse steps", "mean-reverting denoising"], "#ffffff", PALETTE["purple_stroke"]),
        Box(920, 1038, 390, 130, "Reuse routed denoiser", ["semantic_mask and M", "forwarded at every sampling step"], "#ffffff", PALETTE["purple_stroke"], body_size=20),
        Box(1370, 1038, 370, 130, "Restored image", ["EMA output", "seam-suppressed cloud-free result"], "#ffffff", PALETTE["purple_stroke"], body_size=20),
        Box(1800, 1038, 450, 160, "Downstream usage", ["visual assessment", "object detection / interpretation", "remote-sensing analysis"], "#ffffff", PALETTE["purple_stroke"], body_size=20),
    ]
    for b in infer_boxes:
        fig.box(b)
    fig.arrow([(860, 1103), (920, 1103)])
    fig.arrow([(1310, 1103), (1370, 1103)])
    fig.arrow([(1740, 1103), (1800, 1103)])
    fig.arrow([(1070, 800), (1070, 930), (710, 930), (710, 1038)])
    add_footer(fig, "Generated from current code: patch_synthetic_clouds_semantic_routed.yaml and semantic_routed_denoiser.py")
    fig.save("semantic_routed_topconf_overview")


def build_detail() -> None:
    fig = FigureBuilder(
        2200,
        1500,
        "Detailed Semantic Routing, Shared Background Prediction, and Overlap Fusion",
        "The heavy branch does not receive global features explicitly; it is fused with a shared whole-image background prediction at the output patch level",
    )
    fig.panel(40, 180, 2120, 1280, "Internal Computation of the Proposed Denoiser", PALETTE["panel"], PALETTE["accent"])

    fig.tag(620, 246, "Innovation A  semantic-guided routing", PALETTE["blue_fill"], PALETTE["blue_stroke"])
    fig.tag(955, 246, "Innovation B  shared background prediction", PALETTE["orange_fill"], PALETTE["orange_stroke"])
    fig.tag(1350, 246, "Innovation C  stride=64 overlap merge", PALETTE["purple_fill"], PALETTE["purple_stroke"])

    fig.panel(90, 330, 420, 1030, "Routing Signals", PALETTE["green_fill"], PALETTE["green_stroke"])
    for b in [
        Box(130, 410, 340, 110, "Input tensor", ["x_t concatenated with cond_image", "6-channel denoiser input"], "#ffffff", PALETTE["green_stroke"], body_size=18),
        Box(130, 555, 340, 105, "Semantic mask", ["semantic_mask", "semantic_mask_available"], "#ffffff", PALETTE["green_stroke"], body_size=18),
        Box(130, 695, 340, 95, "Cloud mask", ["M"], "#ffffff", PALETTE["green_stroke"], body_size=18),
        Box(130, 825, 340, 110, "Border handling", ["exclude padded borders", "keep real pixels"], "#ffffff", PALETTE["green_stroke"], body_size=18),
        Box(130, 970, 340, 170, "Route-map construction", ["union(semantic_mask, M)", "patch score: max > 0.005"], "#f8fff9", PALETTE["green_stroke"], body_size=18),
    ]:
        fig.box(b)
    fig.arrow([(300, 520), (300, 555)])
    fig.arrow([(300, 660), (300, 695)])
    fig.arrow([(300, 790), (300, 825)])
    fig.arrow([(300, 935), (300, 970)])

    fig.panel(560, 330, 890, 1030, "Dual-Branch Routed Denoising", PALETTE["blue_fill"], PALETTE["blue_stroke"])
    branch_boxes = [
        Box(610, 405, 270, 120, "Patchify", ["route_patch_size = 128", "merge_stride = 64"], "#ffffff", PALETTE["blue_stroke"], body_size=19),
        Box(920, 405, 250, 120, "Light branch", ["whole-image prediction", "lightweight background denoiser"], "#fff3eb", PALETTE["orange_stroke"], body_size=19),
        Box(1210, 405, 200, 120, "Heavy branch", ["ImageTransformer", "routed patches only"], "#f6efff", PALETTE["purple_stroke"], body_size=19),
        Box(670, 600, 670, 180, "Shared background prediction", ["B = F_l(x)", "the whole-image background result is patchified", "selected routed patches use B_i as the fusion base"], "#ffffff", PALETTE["blue_stroke"], title_size=28, body_size=19),
        Box(670, 835, 670, 210, "Patch-level smooth fusion", ["For routed patch i:", "P_hat_i = A ⊙ H_i + (1 - A) ⊙ B_i", "A is a smooth blend window controlled by", "heavy_patch_blend_margin = 16"], "#ffffff", PALETTE["blue_stroke"], title_size=28, body_size=19),
        Box(670, 1100, 670, 160, "Overlap reconstruction", ["fold fused patches back to the image", "average by overlap count", "crop padding and output final restoration"], "#fbfcff", PALETTE["blue_stroke"], title_size=28, body_size=19),
    ]
    for b in branch_boxes:
        fig.box(b)
    fig.arrow([(880, 465), (920, 465)])
    fig.arrow([(880, 465), (1210, 465)])
    fig.arrow([(1045, 525), (1045, 560), (1005, 560), (1005, 600)])
    fig.arrow([(1310, 525), (1310, 835)])
    fig.arrow([(1005, 780), (1005, 835)])
    fig.arrow([(1005, 1045), (1005, 1100)])
    fig.arrow([(510, 1055), (610, 1055), (610, 465)])

    fig.panel(1500, 330, 610, 1030, "Training Supervision", PALETTE["orange_fill"], PALETTE["orange_stroke"])
    sup_boxes = [
        Box(1545, 405, 520, 140, "Residual diffusion supervision", ["x_t = x_0 + mean-reverting term + sigma * noise", "current implementation uses EDM sigma-to-state mapping"], "#ffffff", PALETTE["orange_stroke"], body_size=18),
        Box(1545, 590, 520, 140, "Importance-aware pixel loss", ["higher weight on cloud regions", "padded borders are ignored"], "#ffffff", PALETTE["orange_stroke"], body_size=18),
        Box(1545, 775, 520, 180, "Asymmetric semantic feature matching", ["foreground_weight = 1.0", "background_weight = 0.10", "cloud_foreground_boost = 1.5", "cloud_background_boost = 0.25"], "#ffffff", PALETTE["orange_stroke"], body_size=18),
        Box(1545, 1010, 520, 135, "Interpretation note", ["This module is shared-background prediction", "not internal feature injection into the heavy branch"], "#fffaf5", PALETTE["orange_stroke"], body_size=18),
    ]
    for b in sup_boxes:
        fig.box(b)
    fig.arrow([(1340, 940), (1440, 940), (1440, 660), (1545, 660)])
    fig.arrow([(1340, 940), (1440, 940), (1440, 865), (1545, 865)])
    fig.arrow([(1340, 1180), (1440, 1180), (1440, 1078), (1545, 1078)])
    add_footer(fig, "Terminology aligned with current implementation on 2026-07-26")
    fig.save("semantic_routed_topconf_detail")


def main() -> None:
    build_overview()
    build_detail()


if __name__ == "__main__":
    main()
