from __future__ import annotations

from pathlib import Path

from PIL import Image

CANVAS_W = 3840
CANVAS_H = 2160
GAP_Y = 0
BG_COLOR = (0, 0, 0)

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "clusters": ROOT / "__docs__" / "clusters.png",
    "rings": ROOT / "__docs__" / "rings.png",
    "swiss": ROOT / "__docs__" / "swiss.png",
    "mesh": ROOT / "__docs__" / "mesh.png",
    "bunny": ROOT / "__docs__" / "bunny.png",
    "duck": ROOT / "__docs__" / "duck.png",
    "vader": ROOT / "__docs__" / "vader.png",
}

OUTPUT = ROOT / "__docs__" / "article" / "cover.png"


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def scale_to_width(img: Image.Image, target_w: int) -> Image.Image:
    if target_w <= 0:
        raise ValueError("target_w must be positive")
    w, h = img.size
    scale = target_w / w
    target_h = max(1, int(h * scale))
    return img.resize((target_w, target_h), Image.LANCZOS)


def compute_layout() -> dict[str, int]:
    row_w = int(CANVAS_W * 0.32)
    mesh_w = int(CANVAS_W * 0.96)

    row_imgs = [
        scale_to_width(load_rgb(SOURCES["clusters"]), row_w),
        scale_to_width(load_rgb(SOURCES["rings"]), row_w),
        scale_to_width(load_rgb(SOURCES["swiss"]), row_w),
    ]
    mesh_img = scale_to_width(load_rgb(SOURCES["mesh"]), mesh_w)
    row2_imgs = [
        scale_to_width(load_rgb(SOURCES["bunny"]), row_w),
        scale_to_width(load_rgb(SOURCES["duck"]), row_w),
        scale_to_width(load_rgb(SOURCES["vader"]), row_w),
    ]

    row1_h = max(img.size[1] for img in row_imgs)
    row3_h = max(img.size[1] for img in row2_imgs)
    total_h = row1_h + mesh_img.size[1] + row3_h + 2 * GAP_Y

    if total_h > CANVAS_H:
        scale = (CANVAS_H - 2 * GAP_Y) / (row1_h + mesh_img.size[1] + row3_h)
        row_w = max(1, int(row_w * scale))
        mesh_w = max(1, int(mesh_w * scale))
        row_imgs = [
            scale_to_width(load_rgb(SOURCES["clusters"]), row_w),
            scale_to_width(load_rgb(SOURCES["rings"]), row_w),
            scale_to_width(load_rgb(SOURCES["swiss"]), row_w),
        ]
        mesh_img = scale_to_width(load_rgb(SOURCES["mesh"]), mesh_w)
        row2_imgs = [
            scale_to_width(load_rgb(SOURCES["bunny"]), row_w),
            scale_to_width(load_rgb(SOURCES["duck"]), row_w),
            scale_to_width(load_rgb(SOURCES["vader"]), row_w),
        ]
        row1_h = max(img.size[1] for img in row_imgs)
        row3_h = max(img.size[1] for img in row2_imgs)

    return {
        "row_w": row_w,
        "mesh_w": mesh_w,
        "row1_h": row1_h,
        "row3_h": row3_h,
        "row_imgs": row_imgs,
        "mesh_img": mesh_img,
        "row2_imgs": row2_imgs,
    }


def main() -> None:
    layout = compute_layout()
    row_w = layout["row_w"]
    mesh_w = layout["mesh_w"]
    row1_h = layout["row1_h"]
    row3_h = layout["row3_h"]
    row_imgs = layout["row_imgs"]
    mesh_img = layout["mesh_img"]
    row2_imgs = layout["row2_imgs"]

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)

    left_margin = max(0, (CANVAS_W - 3 * row_w) // 2)
    y = max(0, (CANVAS_H - (row1_h + mesh_img.size[1] + row3_h + 2 * GAP_Y)) // 2)
    if CANVAS_W == 3 * row_w:
        left_margin = 0
    if (row1_h + mesh_img.size[1] + row3_h + 2 * GAP_Y) >= CANVAS_H:
        y = 0
    for idx, img in enumerate(row_imgs):
        x = left_margin + idx * row_w
        y_offset = y + (row1_h - img.size[1]) // 2
        canvas.paste(img, (x, y_offset))

    y += row1_h + GAP_Y
    mesh_x = (CANVAS_W - mesh_w) // 2
    canvas.paste(mesh_img, (mesh_x, y))

    y += mesh_img.size[1] + GAP_Y
    for idx, img in enumerate(row2_imgs):
        x = left_margin + idx * row_w
        y_offset = y + (row3_h - img.size[1]) // 2
        canvas.paste(img, (x, y_offset))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
