from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2200
HEIGHT = 1460


def rect(x, y, w, h, fill, stroke, text_lines, font_size=24, rx=18, ry=18, text_color="#102020", stroke_width=2):
    lines = []
    lines.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{ry}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )
    if text_lines:
        total_h = len(text_lines) * (font_size + 6)
        start_y = y + h / 2 - total_h / 2 + font_size
        for i, line in enumerate(text_lines):
            lines.append(
                f'<text x="{x + w / 2}" y="{start_y + i * (font_size + 6)}" '
                f'font-size="{font_size}" text-anchor="middle" '
                f'font-family="Microsoft YaHei, Segoe UI, sans-serif" '
                f'fill="{text_color}" font-weight="600">{escape(line)}</text>'
            )
    return "\n".join(lines)


def label(x, y, text, font_size=28, color="#17303a", weight="700", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-size="{font_size}" text-anchor="{anchor}" '
        f'font-family="Microsoft YaHei, Segoe UI, sans-serif" fill="{color}" '
        f'font-weight="{weight}">{escape(text)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#38566b", width=4, dashed=False):
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'marker-end="url(#arrow)"{dash}/>'
    )


def connector(points, color="#38566b", width=4, dashed=False):
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" '
        f'marker-end="url(#arrow)"{dash}/>'
    )


def section_frame(x, y, w, h, title, fill, stroke):
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" ry="28" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>',
            label(x + 26, y + 42, title, font_size=30, color=stroke),
        ]
    )


def build_svg() -> str:
    parts = []
    parts.append(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#f4efe5"/>
    <stop offset="45%" stop-color="#eef6fb"/>
    <stop offset="100%" stop-color="#f8fbf4"/>
  </linearGradient>
  <linearGradient id="titlebar" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#1f516d"/>
    <stop offset="100%" stop-color="#497d62"/>
  </linearGradient>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#38566b"/>
  </marker>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#203040" flood-opacity="0.12"/>
  </filter>
</defs>
<rect width="100%" height="100%" fill="url(#bg)"/>
<rect x="34" y="28" width="{WIDTH - 68}" height="96" rx="28" ry="28" fill="url(#titlebar)"/>
'''
    )

    parts.append(label(70, 88, "语义路由残差扩散去云框架图", font_size=42, color="#ffffff", weight="800"))
    parts.append(label(70, 118, "配置源: configs/example_training/patch_synthetic_clouds_semantic_routed.yaml", font_size=20, color="#dfeff7", weight="500"))

    # Left panel
    lx, ly, lw, lh = 40, 150, 1260, 1260
    parts.append(section_frame(lx, ly, lw, lh, "A. 端到端训练与采样链路", "#fcfdfd", "#345c71"))

    # Right panel
    rx, ry, rw, rh = 1330, 150, 830, 1260
    parts.append(section_frame(rx, ry, rw, rh, "B. Semantic Routed Patch Denoiser 细节", "#fffdf8", "#8a5a2b"))

    # Configuration strip
    parts.append(rect(90, 220, 1160, 88, "#f3ebd8", "#a48145", ["实验配置: ResidualDiffusionEngine + CloudRemovalPatchDataset + SemanticRoutedPatchDenoiser"], font_size=24, text_color="#402f12"))

    # Data section
    parts.append(section_frame(80, 350, 290, 520, "数据层", "#edf7f2", "#2f7a68"))
    parts.append(rect(108, 410, 234, 82, "#ffffff", "#2f7a68", ["CloudRemoval", "PatchDataset"], font_size=22))
    parts.append(rect(108, 525, 234, 68, "#f8fffd", "#4b9a84", ["cond_image", "有云输入 / μ"], font_size=22))
    parts.append(rect(108, 615, 234, 68, "#f8fffd", "#4b9a84", ["label", "无云真值"], font_size=22))
    parts.append(rect(108, 705, 234, 68, "#f8fffd", "#4b9a84", ["M", "云掩码"], font_size=22))
    parts.append(rect(108, 795, 234, 68, "#f8fffd", "#4b9a84", ["semantic_mask", "semantic_mask_available"], font_size=20))
    parts.append(arrow(225, 492, 225, 525, color="#2f7a68"))
    parts.append(arrow(225, 492, 225, 615, color="#2f7a68"))
    parts.append(arrow(225, 492, 225, 705, color="#2f7a68"))
    parts.append(arrow(225, 492, 225, 795, color="#2f7a68"))

    # Engine section
    parts.append(section_frame(405, 350, 800, 700, "训练主干", "#eef3fb", "#4966b4"))
    parts.append(rect(450, 410, 255, 84, "#ffffff", "#4966b4", ["ResidualDiffusion", "Engine"], font_size=22))
    parts.append(rect(760, 410, 210, 84, "#fbfdff", "#5a7dd1", ["IdentityFirstStage", "编码 label"], font_size=22))
    parts.append(rect(995, 410, 170, 84, "#fbfdff", "#5a7dd1", ["IdentityFirstStage", "编码 μ"], font_size=22))
    parts.append(rect(450, 540, 255, 82, "#ffffff", "#4966b4", ["GeneralConditioner", "IdentityEmbedder(cond_image)"], font_size=21))
    parts.append(rect(760, 540, 190, 82, "#ffffff", "#4966b4", ["EDMSigma2St", "alpha = 3.0"], font_size=23))
    parts.append(rect(985, 540, 180, 82, "#ffffff", "#4966b4", ["ResidualDenoiser", "ResidualEDMScaling"], font_size=21))
    parts.append(rect(450, 675, 245, 82, "#ffffff", "#4966b4", ["CloudRemoval", "Wrapper", "concat(x_t, cond_image)"], font_size=19))
    parts.append(rect(715, 650, 450, 122, "#f6f9ff", "#3f67c8", ["SemanticRoutedPatchDenoiser", "6通道输入", "patch=128, mode=max, threshold=0.005"], font_size=20))
    parts.append(rect(430, 820, 360, 102, "#fff4ee", "#c76434", ["SemanticFeatureMatching", "ResidualDiffusionLoss", "像素重建损失 + 语义特征匹配损失"], font_size=20, text_color="#451c0c"))
    parts.append(rect(810, 820, 355, 102, "#fff4ee", "#c76434", ["MaskWeightedAsymmetric", "FeatureMatchingLoss", "VGG relu2_2 / relu3_3 / relu4_3"], font_size=19, text_color="#451c0c"))
    parts.append(rect(585, 950, 330, 78, "#fff8ef", "#ba6b2d", ["总损失 = cloud-aware pixel loss + 0.10 × feature matching"], font_size=22, text_color="#49270b"))

    # Engine arrows
    parts.append(connector([(342, 560), (395, 560), (395, 582), (450, 582)], color="#2f7a68"))
    parts.append(connector([(342, 650), (395, 650), (395, 452), (760, 452)], color="#2f7a68"))
    parts.append(connector([(342, 560), (395, 560), (395, 452), (995, 452)], color="#2f7a68"))
    parts.append(connector([(342, 740), (395, 740), (395, 866), (450, 866)], color="#2f7a68"))
    parts.append(connector([(342, 830), (395, 830), (395, 882), (450, 882)], color="#2f7a68"))
    parts.append(connector([(705, 582), (985, 582)], color="#4966b4"))
    parts.append(connector([(950, 582), (950, 582), (950, 582)], color="#4966b4"))
    parts.append(connector([(1080, 494), (1080, 540)], color="#4966b4"))
    parts.append(connector([(705, 452), (760, 452)], color="#4966b4"))
    parts.append(connector([(705, 452), (995, 452)], color="#4966b4"))
    parts.append(connector([(1075, 622), (1075, 660)], color="#4966b4"))
    parts.append(connector([(572, 622), (572, 675)], color="#4966b4"))
    parts.append(connector([(695, 716), (735, 716)], color="#4966b4"))
    parts.append(connector([(1165, 716), (1165, 866), (795, 866)], color="#4966b4"))
    parts.append(connector([(950, 772), (950, 866), (840, 866)], color="#4966b4"))
    parts.append(connector([(795, 958), (795, 989), (585, 989)], color="#c76434"))
    parts.append(connector([(840, 866), (840, 989), (915, 989)], color="#c76434"))

    # Sampler section
    parts.append(section_frame(405, 1085, 800, 270, "采样 / 验证 / 预测", "#f6fbf8", "#4f7d61"))
    parts.append(rect(450, 1145, 245, 82, "#ffffff", "#4f7d61", ["ResidualHeunEDMSampler", "8步反向扩散"], font_size=23))
    parts.append(rect(735, 1145, 220, 82, "#ffffff", "#4f7d61", ["重复调用", "ResidualDenoiser"], font_size=24))
    parts.append(rect(990, 1145, 175, 82, "#ffffff", "#4f7d61", ["解码输出", "restored image"], font_size=24))
    parts.append(rect(735, 1260, 430, 60, "#f8fffd", "#6da286", ["采样时继续传入 semantic_mask / semantic_mask_available / M, 保持与训练一致的路由行为"], font_size=19))
    parts.append(connector([(572, 1028), (572, 1186), (450, 1186)], color="#4f7d61", dashed=True))
    parts.append(connector([(695, 1186), (735, 1186)], color="#4f7d61"))
    parts.append(connector([(955, 1186), (985, 1186), (985, 716), (1165, 716)], color="#4f7d61", dashed=True))
    parts.append(connector([(1165, 1186), (1195, 1186)], color="#4f7d61"))

    # Notes
    parts.append(rect(88, 900, 255, 98, "#fcfffe", "#69a08e", ["数据集同时提供:", "条件图、目标图、云掩码、语义掩码"], font_size=21, text_color="#16332c"))
    parts.append(rect(88, 1015, 255, 120, "#fcfffe", "#69a08e", ["cond_image 有双重角色:", "1. 作为 mean_key 的 μ", "2. 作为 concat 条件输入"], font_size=20, text_color="#16332c"))

    # Right panel routed denoiser
    parts.append(rect(1370, 230, 750, 74, "#f3e6d8", "#a6682e", ["输入: x_t(3通道) 与 cond_image(3通道) 拼接后形成 6通道特征"], font_size=24, text_color="#43230b"))
    parts.append(rect(1390, 350, 230, 82, "#eef9f0", "#3d8b4b", ["semantic_mask"], font_size=24))
    parts.append(rect(1660, 350, 230, 82, "#eef9f0", "#3d8b4b", ["cloud mask M"], font_size=24))
    parts.append(rect(1525, 470, 230, 92, "#f4fbf5", "#5d9a69", ["组合路由掩码", "combine_mode = union"], font_size=23))
    parts.append(rect(1525, 615, 230, 92, "#eef6ff", "#3f78d1", ["Pad 到 128 的整数倍", "Patchify 为 128×128 块"], font_size=22))
    parts.append(rect(1525, 760, 230, 104, "#eef6ff", "#3f78d1", ["计算 patch route score", "route_mode = max", "score > 0.005 进入重分支"], font_size=22))

    parts.append(rect(1390, 915, 260, 140, "#fff3eb", "#c66a2b", ["轻分支", "LightweightBackground", "Denoiser", "2层卷积, hidden=32"], font_size=21, text_color="#32190a"))
    parts.append(rect(1720, 900, 340, 170, "#fff3eb", "#c66a2b", ["重分支 Backbone", "ImageTransformerDenoiser", "ModelInterface", "宽度: 64 / 128 / 256 / 512", "注意力: 邻域, 邻域, 全局, 全局"], font_size=19, text_color="#32190a"))

    parts.append(rect(1510, 1135, 280, 86, "#f8fafc", "#5b6c7c", ["用重分支结果覆盖被路由的 patch"], font_size=23))
    parts.append(rect(1510, 1270, 280, 76, "#f8fafc", "#5b6c7c", ["Unpatchify + 去除 padding"], font_size=23))
    parts.append(rect(1840, 1265, 250, 86, "#f7f1fb", "#8a5cb8", ["输出: 最终预测残差 / 去噪结果"], font_size=23, text_color="#241230"))

    parts.append(connector([(1505, 304), (1505, 350)], color="#a6682e"))
    parts.append(connector([(1780, 304), (1780, 350)], color="#a6682e"))
    parts.append(connector([(1505, 432), (1505, 470), (1640, 470)], color="#3d8b4b"))
    parts.append(connector([(1775, 432), (1775, 470), (1640, 470)], color="#3d8b4b"))
    parts.append(connector([(1640, 562), (1640, 615)], color="#3f78d1"))
    parts.append(connector([(1640, 707), (1640, 760)], color="#3f78d1"))
    parts.append(connector([(1640, 864), (1640, 995), (1650, 995)], color="#3f78d1"))
    parts.append(connector([(1640, 864), (1640, 995), (1740, 995)], color="#3f78d1"))
    parts.append(connector([(1650, 995), (1650, 1178), (1510, 1178)], color="#c66a2b"))
    parts.append(connector([(1900, 1070), (1900, 1178), (1790, 1178)], color="#c66a2b"))
    parts.append(connector([(1650, 1221), (1650, 1270)], color="#5b6c7c"))
    parts.append(connector([(1790, 1308), (1840, 1308)], color="#5b6c7c"))

    parts.append(label(1410, 1395, "图中所有模块关系均来自当前仓库实现，而非概念性推断。", font_size=20, color="#5a6470", weight="500"))
    parts.append("</svg>")
    return "\n".join(parts)


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/msyhbd.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/msyh.ttc",
            ]
        )
    else:
        candidates.extend(
            [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simsun.ttc",
                "C:/Windows/Fonts/arial.ttf",
            ]
        )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_vertical_gradient(draw: ImageDraw.ImageDraw, width: int, height: int, top, bottom):
    for y in range(height):
        t = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width, y), fill=color)


def draw_round_box(draw, x, y, w, h, fill, outline, radius=18, width=2):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)


def draw_center_text(draw, box, lines, font_size=24, fill=(16, 32, 32), bold=True, line_gap=6):
    font = load_font(font_size, bold=bold)
    x, y, w, h = box
    total_h = 0
    metrics = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lh = bbox[3] - bbox[1]
        metrics.append((line, bbox, lh))
        total_h += lh
    total_h += line_gap * (len(lines) - 1)
    cy = y + (h - total_h) / 2
    for line, bbox, lh in metrics:
        lw = bbox[2] - bbox[0]
        draw.text((x + w / 2 - lw / 2, cy), line, font=font, fill=fill)
        cy += lh + line_gap


def draw_left_text(draw, x, y, text, font_size=28, fill=(23, 48, 58), bold=True):
    font = load_font(font_size, bold=bold)
    draw.text((x, y), text, font=font, fill=fill)


def draw_arrow(draw, points, fill=(56, 86, 107), width=4, dashed=False):
    if dashed:
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            length = max((dx * dx + dy * dy) ** 0.5, 1.0)
            ux = dx / length
            uy = dy / length
            dash_len = 12
            gap = 8
            pos = 0.0
            while pos < length:
                end = min(pos + dash_len, length)
                sx = x1 + ux * pos
                sy = y1 + uy * pos
                ex = x1 + ux * end
                ey = y1 + uy * end
                draw.line((sx, sy, ex, ey), fill=fill, width=width)
                pos += dash_len + gap
    else:
        draw.line(points, fill=fill, width=width)

    if len(points) >= 2:
        x1, y1 = points[-2]
        x2, y2 = points[-1]
        dx = x2 - x1
        dy = y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        arrow_len = 16
        arrow_w = 8
        p1 = (x2, y2)
        p2 = (x2 - ux * arrow_len + px * arrow_w, y2 - uy * arrow_len + py * arrow_w)
        p3 = (x2 - ux * arrow_len - px * arrow_w, y2 - uy * arrow_len - py * arrow_w)
        draw.polygon([p1, p2, p3], fill=fill)


def render_png(path: Path):
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw_vertical_gradient(draw, WIDTH, HEIGHT, (244, 239, 229), (240, 248, 244))

    draw_round_box(draw, 34, 28, WIDTH - 68, 96, (31, 81, 109), (31, 81, 109), radius=28, width=1)
    draw_left_text(draw, 70, 46, "语义路由残差扩散去云框架图", font_size=42, fill=(255, 255, 255), bold=True)
    draw_left_text(draw, 70, 92, "配置源: configs/example_training/patch_synthetic_clouds_semantic_routed.yaml", font_size=20, fill=(223, 239, 247), bold=False)

    draw_round_box(draw, 40, 150, 1260, 1260, (252, 253, 253), (52, 92, 113), radius=28, width=3)
    draw_left_text(draw, 66, 168, "A. 端到端训练与采样链路", font_size=30, fill=(52, 92, 113), bold=True)
    draw_round_box(draw, 1330, 150, 830, 1260, (255, 253, 248), (138, 90, 43), radius=28, width=3)
    draw_left_text(draw, 1356, 168, "B. Semantic Routed Patch Denoiser 细节", font_size=30, fill=(138, 90, 43), bold=True)

    draw_round_box(draw, 90, 220, 1160, 88, (243, 235, 216), (164, 129, 69), radius=18)
    draw_center_text(draw, (90, 220, 1160, 88), ["实验配置: ResidualDiffusionEngine + CloudRemovalPatchDataset + SemanticRoutedPatchDenoiser"], font_size=24, fill=(64, 47, 18))

    draw_round_box(draw, 80, 350, 290, 520, (237, 247, 242), (47, 122, 104), radius=28, width=3)
    draw_left_text(draw, 106, 368, "数据层", font_size=30, fill=(47, 122, 104), bold=True)
    draw_round_box(draw, 108, 410, 234, 82, (255, 255, 255), (47, 122, 104))
    draw_center_text(draw, (108, 410, 234, 82), ["CloudRemoval", "PatchDataset"], font_size=22)
    for y, lines, size in [
        (525, ["cond_image", "有云输入 / μ"], 22),
        (615, ["label", "无云真值"], 22),
        (705, ["M", "云掩码"], 22),
        (795, ["semantic_mask", "semantic_mask_available"], 20),
    ]:
        draw_round_box(draw, 108, y, 234, 68, (248, 255, 253), (75, 154, 132))
        draw_center_text(draw, (108, y, 234, 68), lines, font_size=size)
    for y in [525, 615, 705, 795]:
        draw_arrow(draw, [(225, 492), (225, y)], fill=(47, 122, 104), width=3)

    draw_round_box(draw, 405, 350, 800, 700, (238, 243, 251), (73, 102, 180), radius=28, width=3)
    draw_left_text(draw, 431, 368, "训练主干", font_size=30, fill=(73, 102, 180), bold=True)
    boxes = [
        (450, 410, 255, 84, (255, 255, 255), (73, 102, 180), ["ResidualDiffusion", "Engine"], 22, (16, 32, 32)),
        (760, 410, 210, 84, (251, 253, 255), (90, 125, 209), ["IdentityFirstStage", "编码 label"], 22, (16, 32, 32)),
        (995, 410, 170, 84, (251, 253, 255), (90, 125, 209), ["IdentityFirstStage", "编码 μ"], 22, (16, 32, 32)),
        (450, 540, 255, 82, (255, 255, 255), (73, 102, 180), ["GeneralConditioner", "IdentityEmbedder(cond_image)"], 21, (16, 32, 32)),
        (760, 540, 190, 82, (255, 255, 255), (73, 102, 180), ["EDMSigma2St", "alpha = 3.0"], 23, (16, 32, 32)),
        (985, 540, 180, 82, (255, 255, 255), (73, 102, 180), ["ResidualDenoiser", "ResidualEDMScaling"], 21, (16, 32, 32)),
        (450, 675, 245, 82, (255, 255, 255), (73, 102, 180), ["CloudRemoval", "Wrapper", "concat(x_t, cond_image)"], 19, (16, 32, 32)),
        (715, 650, 450, 122, (246, 249, 255), (63, 103, 200), ["SemanticRoutedPatchDenoiser", "6通道输入", "patch=128, mode=max, threshold=0.005"], 20, (16, 32, 32)),
        (430, 820, 360, 102, (255, 244, 238), (199, 100, 52), ["SemanticFeatureMatching", "ResidualDiffusionLoss", "像素重建损失 + 语义特征匹配损失"], 20, (69, 28, 12)),
        (810, 820, 355, 102, (255, 244, 238), (199, 100, 52), ["MaskWeightedAsymmetric", "FeatureMatchingLoss", "VGG relu2_2 / relu3_3 / relu4_3"], 19, (69, 28, 12)),
        (585, 950, 330, 78, (255, 248, 239), (186, 107, 45), ["总损失 = cloud-aware pixel loss + 0.10 × feature matching"], 22, (73, 39, 11)),
    ]
    for x, y, w, h, fill, stroke, lines, size, text_fill in boxes:
        draw_round_box(draw, x, y, w, h, fill, stroke)
        draw_center_text(draw, (x, y, w, h), lines, font_size=size, fill=text_fill)

    for pts, color, dashed in [
        ([(342, 560), (395, 560), (395, 582), (450, 582)], (47, 122, 104), False),
        ([(342, 650), (395, 650), (395, 452), (760, 452)], (47, 122, 104), False),
        ([(342, 560), (395, 560), (395, 452), (995, 452)], (47, 122, 104), False),
        ([(342, 740), (395, 740), (395, 866), (450, 866)], (47, 122, 104), False),
        ([(342, 830), (395, 830), (395, 882), (450, 882)], (47, 122, 104), False),
        ([(705, 582), (985, 582)], (73, 102, 180), False),
        ([(1080, 494), (1080, 540)], (73, 102, 180), False),
        ([(705, 452), (760, 452)], (73, 102, 180), False),
        ([(705, 452), (995, 452)], (73, 102, 180), False),
        ([(1075, 622), (1075, 660)], (73, 102, 180), False),
        ([(572, 622), (572, 675)], (73, 102, 180), False),
        ([(695, 716), (735, 716)], (73, 102, 180), False),
        ([(1165, 716), (1165, 866), (795, 866)], (73, 102, 180), False),
        ([(950, 772), (950, 866), (840, 866)], (73, 102, 180), False),
        ([(795, 958), (795, 989), (585, 989)], (199, 100, 52), False),
        ([(840, 866), (840, 989), (915, 989)], (199, 100, 52), False),
    ]:
        draw_arrow(draw, pts, fill=color, width=4, dashed=dashed)

    draw_round_box(draw, 405, 1085, 800, 270, (246, 251, 248), (79, 125, 97), radius=28, width=3)
    draw_left_text(draw, 431, 1103, "采样 / 验证 / 预测", font_size=30, fill=(79, 125, 97), bold=True)
    for x, y, w, h, lines, size in [
        (450, 1145, 245, 82, ["ResidualHeunEDMSampler", "8步反向扩散"], 23),
        (735, 1145, 220, 82, ["重复调用", "ResidualDenoiser"], 24),
        (990, 1145, 175, 82, ["解码输出", "restored image"], 24),
    ]:
        draw_round_box(draw, x, y, w, h, (255, 255, 255), (79, 125, 97))
        draw_center_text(draw, (x, y, w, h), lines, font_size=size)
    draw_round_box(draw, 735, 1260, 430, 60, (248, 255, 253), (109, 162, 134))
    draw_center_text(draw, (735, 1260, 430, 60), ["采样时继续传入 semantic_mask / semantic_mask_available / M, 保持与训练一致的路由行为"], font_size=19)
    draw_arrow(draw, [(572, 1028), (572, 1186), (450, 1186)], fill=(79, 125, 97), width=4, dashed=True)
    draw_arrow(draw, [(695, 1186), (735, 1186)], fill=(79, 125, 97), width=4)
    draw_arrow(draw, [(955, 1186), (985, 1186), (985, 716), (1165, 716)], fill=(79, 125, 97), width=4, dashed=True)
    draw_arrow(draw, [(1165, 1186), (1195, 1186)], fill=(79, 125, 97), width=4)

    draw_round_box(draw, 88, 900, 255, 98, (252, 255, 254), (105, 160, 142))
    draw_center_text(draw, (88, 900, 255, 98), ["数据集同时提供:", "条件图、目标图、云掩码、语义掩码"], font_size=21, fill=(22, 51, 44))
    draw_round_box(draw, 88, 1015, 255, 120, (252, 255, 254), (105, 160, 142))
    draw_center_text(draw, (88, 1015, 255, 120), ["cond_image 有双重角色:", "1. 作为 mean_key 的 μ", "2. 作为 concat 条件输入"], font_size=20, fill=(22, 51, 44))

    # Right panel
    draw_round_box(draw, 1370, 230, 750, 74, (243, 230, 216), (166, 104, 46))
    draw_center_text(draw, (1370, 230, 750, 74), ["输入: x_t(3通道) 与 cond_image(3通道) 拼接后形成 6通道特征"], font_size=24, fill=(67, 35, 11))
    for x, y, w, h, fill, stroke, lines, size, text_fill in [
        (1390, 350, 230, 82, (238, 249, 240), (61, 139, 75), ["semantic_mask"], 24, (16, 37, 19)),
        (1660, 350, 230, 82, (238, 249, 240), (61, 139, 75), ["cloud mask M"], 24, (16, 37, 19)),
        (1525, 470, 230, 92, (244, 251, 245), (93, 154, 105), ["组合路由掩码", "combine_mode = union"], 23, (16, 37, 19)),
        (1525, 615, 230, 92, (238, 246, 255), (63, 120, 209), ["Pad 到 128 的整数倍", "Patchify 为 128×128 块"], 22, (14, 29, 57)),
        (1525, 760, 230, 104, (238, 246, 255), (63, 120, 209), ["计算 patch route score", "route_mode = max", "score > 0.005 进入重分支"], 22, (14, 29, 57)),
        (1390, 915, 260, 140, (255, 243, 235), (198, 106, 43), ["轻分支", "LightweightBackground", "Denoiser", "2层卷积, hidden=32"], 21, (50, 25, 10)),
        (1720, 900, 340, 170, (255, 243, 235), (198, 106, 43), ["重分支 Backbone", "ImageTransformerDenoiser", "ModelInterface", "宽度: 64 / 128 / 256 / 512", "注意力: 邻域, 邻域, 全局, 全局"], 19, (50, 25, 10)),
        (1510, 1135, 280, 86, (248, 250, 252), (91, 108, 124), ["用重分支结果覆盖被路由的 patch"], 23, (31, 40, 48)),
        (1510, 1270, 280, 76, (248, 250, 252), (91, 108, 124), ["Unpatchify + 去除 padding"], 23, (31, 40, 48)),
        (1840, 1265, 250, 86, (247, 241, 251), (138, 92, 184), ["输出: 最终预测残差 / 去噪结果"], 23, (36, 18, 48)),
    ]:
        draw_round_box(draw, x, y, w, h, fill, stroke)
        draw_center_text(draw, (x, y, w, h), lines, font_size=size, fill=text_fill)

    for pts, color, dashed in [
        ([(1505, 304), (1505, 350)], (166, 104, 46), False),
        ([(1780, 304), (1780, 350)], (166, 104, 46), False),
        ([(1505, 432), (1505, 470), (1640, 470)], (61, 139, 75), False),
        ([(1775, 432), (1775, 470), (1640, 470)], (61, 139, 75), False),
        ([(1640, 562), (1640, 615)], (63, 120, 209), False),
        ([(1640, 707), (1640, 760)], (63, 120, 209), False),
        ([(1640, 864), (1640, 995), (1650, 995)], (63, 120, 209), False),
        ([(1640, 864), (1640, 995), (1740, 995)], (63, 120, 209), False),
        ([(1650, 995), (1650, 1178), (1510, 1178)], (198, 106, 43), False),
        ([(1900, 1070), (1900, 1178), (1790, 1178)], (198, 106, 43), False),
        ([(1650, 1221), (1650, 1270)], (91, 108, 124), False),
        ([(1790, 1308), (1840, 1308)], (91, 108, 124), False),
    ]:
        draw_arrow(draw, pts, fill=color, width=4, dashed=dashed)

    draw_left_text(draw, 1410, 1395, "图中所有模块关系均来自当前仓库实现，而非概念性推断。", font_size=20, fill=(90, 100, 112), bold=False)
    image.save(path)


def main():
    output_svg = Path("D:/EMRDM-main/docs/semantic_routed_framework_zh.svg")
    output_png = Path("D:/EMRDM-main/docs/semantic_routed_framework_zh.png")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text(build_svg(), encoding="utf-8")
    render_png(output_png)
    print(output_svg)
    print(output_png)


if __name__ == "__main__":
    main()
