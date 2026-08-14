"""Generates small, deterministic sample CSVs matching the brief's schema,
for local testing/demo of the pipeline without needing real e-commerce data.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent

random.seed(7)

PRODUCTS = [
    ("P001", "Aurora Blender X1", "Kitchen"),
    ("P002", "TrailBlaze Backpack", "Outdoor"),
    ("P003", "LumaGlow Desk Lamp", "Home Office"),
    ("P004", "SonicWave Earbuds", "Electronics"),
    ("P005", "IronGrip Yoga Mat", "Fitness"),
    ("P006", "CrispAir Fan", "Home"),
    ("P007", "PixelView Monitor Stand", "Electronics"),
    ("P008", "BrewMaster Kettle", "Kitchen"),
]

GOOD_REVIEWS = [
    "Works great, exactly as described. Very happy with this purchase.",
    "Excellent quality and fast shipping, would buy again.",
    "Solid build, does the job well.",
    "Nice design and easy to use.",
]

BAD_QUALITY_REVIEWS = [
    "Broken after two days of light use, very disappointed.",
    "Poor quality materials, it fell apart within a week.",
    "Stopped working after a month, seems defective.",
    "The item arrived damaged and cracked in the box.",
    "Cheaply made, wore out much faster than expected.",
]

OTHER_COMPLAINTS = [
    "Shipping took forever, item was fine though.",
    "A bit pricey for what you get, but works fine.",
    "Customer service was slow to respond to my question.",
]


def write_products() -> None:
    with open(OUT_DIR / "products.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id", "product_name", "category"])
        writer.writerows(PRODUCTS)


def write_sales(n_sales: int = 300) -> None:
    with open(OUT_DIR / "sales.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sale_id", "product_id", "customer_id", "sale_date", "quantity", "price_per_item"])
        for i in range(1, n_sales + 1):
            product = random.choice(PRODUCTS)
            date_formats = ["2024-{:02d}-{:02d}", "{:02d}/{:02d}/2024", "2024/{:02d}/{:02d}"]
            month, day = random.randint(1, 12), random.randint(1, 28)
            date_str = random.choice(date_formats).format(month, day) if "/" not in date_formats[0] else None
            fmt = random.choice(date_formats)
            date_str = fmt.format(month, day) if fmt != "2024/{:02d}/{:02d}" else fmt.format(month, day)
            qty = random.choice([1, 1, 1, 2, 3, None])  # occasional null to exercise Silver cleansing
            price = round(random.uniform(9.99, 199.99), 2)
            writer.writerow([i, product[0], f"C{random.randint(1000, 1050)}", date_str, qty, price])


def write_reviews(n_reviews: int = 220) -> None:
    with open(OUT_DIR / "reviews.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "product_id", "customer_id", "rating", "review_date", "review_text"])
        for i in range(1, n_reviews + 1):
            # Weight certain products toward more quality complaints for a realistic demo.
            product = PRODUCTS[i % len(PRODUCTS)]
            is_bad_quality_product = product[0] in {"P001", "P006"}
            roll = random.random()
            if is_bad_quality_product and roll < 0.55:
                text = random.choice(BAD_QUALITY_REVIEWS)
                rating = random.choice([1, 2])
            elif roll < 0.15:
                text = random.choice(BAD_QUALITY_REVIEWS)
                rating = random.choice([1, 2])
            elif roll < 0.30:
                text = random.choice(OTHER_COMPLAINTS)
                rating = random.choice([2, 3])
            else:
                text = random.choice(GOOD_REVIEWS)
                rating = random.choice([4, 5, None])  # occasional null rating
            writer.writerow([i, product[0], f"C{random.randint(1000, 1050)}",
                              rating, f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}", text])


if __name__ == "__main__":
    write_products()
    write_sales()
    write_reviews()
    print(f"Sample CSVs written to {OUT_DIR}")
