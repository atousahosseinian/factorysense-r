from pathlib import Path
from PIL import Image, ImageDraw
import random


def make_good_image(size=256):
    image = Image.new("RGB", (size, size), color=(230, 230, 230))
    draw = ImageDraw.Draw(image)

    draw.ellipse((64, 64, 192, 192), fill=(170, 190, 210), outline=(80, 100, 120), width=4)
    draw.rectangle((105, 35, 151, 75), fill=(150, 170, 190), outline=(80, 100, 120), width=3)

    # Small normal texture/noise
    for _ in range(80):
        x = random.randint(70, 185)
        y = random.randint(75, 185)
        shade = random.randint(150, 210)
        draw.point((x, y), fill=(shade, shade, shade))

    return image


def add_defect(image, defect_type):
    image = image.copy()
    mask = Image.new("L", image.size, color=0)

    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    if defect_type == "scratch":
        points = [(85, 130), (115, 118), (145, 138), (175, 120)]
        draw.line(points, fill=(40, 40, 40), width=5)
        mask_draw.line(points, fill=255, width=8)

    elif defect_type == "contamination":
        box = (135, 135, 170, 170)
        draw.ellipse(box, fill=(55, 45, 35))
        mask_draw.ellipse(box, fill=255)

    elif defect_type == "broken_large":
        polygon = [(150, 80), (190, 95), (180, 150), (145, 135)]
        draw.polygon(polygon, fill=(35, 35, 35))
        mask_draw.polygon(polygon, fill=255)

    return image, mask


def main():
    root = Path("data/mvtec")
    category = "bottle"

    train_good = root / category / "train" / "good"
    test_good = root / category / "test" / "good"
    defect_types = ["scratch", "contamination", "broken_large"]

    train_good.mkdir(parents=True, exist_ok=True)
    test_good.mkdir(parents=True, exist_ok=True)

    for defect_type in defect_types:
        (root / category / "test" / defect_type).mkdir(parents=True, exist_ok=True)
        (root / category / "ground_truth" / defect_type).mkdir(parents=True, exist_ok=True)

    for i in range(12):
        image = make_good_image()
        image.save(train_good / f"{i:03d}.png")

    for i in range(6):
        image = make_good_image()
        image.save(test_good / f"{i:03d}.png")

    for defect_type in defect_types:
        for i in range(4):
            image = make_good_image()
            defective, mask = add_defect(image, defect_type)

            defective.save(root / category / "test" / defect_type / f"{i:03d}.png")
            mask.save(root / category / "ground_truth" / defect_type / f"{i:03d}_mask.png")

    print("Demo MVTec-style dataset created at data/mvtec/bottle")


if __name__ == "__main__":
    main()
