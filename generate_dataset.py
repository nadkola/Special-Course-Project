"""
generate_dataset.py — Creates the Global Retail Sales dataset (10,000 transactions)
Date range: January 2023 – December 2025
Includes: seasonal trends, year-over-year growth, realistic regional/category patterns

Dimensions: time, geography, product, customer
Measures: quantity, unit_price, revenue, cost, profit, profit_margin
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 10000

# ── Time Dimension (2024-01-01 to 2025-12-31) ──────────────────
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2025-12-31")
date_range = pd.date_range(start_date, end_date, freq="D")

# Seasonal weights: more transactions in Q4 (holiday season) and Q2
month_weights = {
    1: 0.75, 2: 0.70, 3: 0.85,    # Q1 — slow start
    4: 0.95, 5: 1.00, 6: 1.05,    # Q2 — spring uptick
    7: 0.90, 8: 0.85, 9: 1.00,    # Q3 — summer dip then recovery
    10: 1.10, 11: 1.30, 12: 1.40, # Q4 — holiday peak
}

# Assign weights to each date based on month + slight 2025 growth (8% more transactions)
date_weights = []
for d in date_range:
    w = month_weights[d.month]
    if d.year == 2024:
        w *= 1.06  # 6% YoY growth over 2023
    elif d.year == 2025:
        w *= 1.14  # 14% growth over 2023 (compounded)
    date_weights.append(w)
date_weights = np.array(date_weights)
date_weights /= date_weights.sum()

order_dates = np.random.choice(date_range, size=N, p=date_weights)

# ── Geography Dimension ─────────────────────────────────────────
geo = {
    "North America": ["United States", "Canada", "Mexico"],
    "Europe": ["United Kingdom", "Germany", "France", "Italy", "Spain"],
    "Asia Pacific": ["China", "Japan", "India", "Australia", "South Korea"],
    "Latin America": ["Brazil", "Argentina", "Colombia", "Chile"],
}
region_weights = [0.35, 0.30, 0.25, 0.10]
region_list = list(geo.keys())

regions = []
countries = []
for _ in range(N):
    r = np.random.choice(region_list, p=region_weights)
    c = np.random.choice(geo[r])
    regions.append(r)
    countries.append(c)

# ── Product Dimension ───────────────────────────────────────────
products = {
    "Electronics": {
        "subcategories": ["Phones", "Laptops", "Tablets", "Accessories", "Monitors"],
        "price_range": (50, 1500),
        "cost_ratio": (0.55, 0.70),
        "weight": 0.30,
    },
    "Furniture": {
        "subcategories": ["Desks", "Chairs", "Bookcases", "Tables", "Storage"],
        "price_range": (30, 800),
        "cost_ratio": (0.50, 0.65),
        "weight": 0.25,
    },
    "Office Supplies": {
        "subcategories": ["Paper", "Binders", "Pens", "Labels", "Envelopes"],
        "price_range": (2, 80),
        "cost_ratio": (0.35, 0.55),
        "weight": 0.25,
    },
    "Clothing": {
        "subcategories": ["Shirts", "Pants", "Jackets", "Shoes", "Accessories"],
        "price_range": (10, 300),
        "cost_ratio": (0.40, 0.60),
        "weight": 0.20,
    },
}
cat_list = list(products.keys())
cat_weights = [products[c]["weight"] for c in cat_list]

categories = []
subcategories = []
unit_prices = []
costs = []

for i in range(N):
    cat = np.random.choice(cat_list, p=cat_weights)
    sub = np.random.choice(products[cat]["subcategories"])
    lo, hi = products[cat]["price_range"]
    price = round(np.random.uniform(lo, hi), 2)

    # Progressive price inflation
    if pd.Timestamp(order_dates[i]).year == 2024:
        price = round(price * 1.03, 2)   # 3% inflation over 2023
    elif pd.Timestamp(order_dates[i]).year == 2025:
        price = round(price * 1.07, 2)   # 7% cumulative inflation over 2023

    cr_lo, cr_hi = products[cat]["cost_ratio"]
    cost = round(price * np.random.uniform(cr_lo, cr_hi), 2)

    categories.append(cat)
    subcategories.append(sub)
    unit_prices.append(price)
    costs.append(cost)

# ── Customer Dimension ──────────────────────────────────────────
segments = ["Consumer", "Corporate", "Home Office"]
seg_weights = [0.50, 0.30, 0.20]
customer_segments = np.random.choice(segments, size=N, p=seg_weights)

# ── Measures ────────────────────────────────────────────────────
quantities = np.random.randint(1, 15, size=N)
unit_prices = np.array(unit_prices)
costs = np.array(costs)
revenues = np.round(quantities * unit_prices, 2)
total_costs = np.round(quantities * costs, 2)
profits = np.round(revenues - total_costs, 2)
profit_margins = np.round(np.where(revenues > 0, (profits / revenues) * 100, 0), 2)

# ── Assemble DataFrame ──────────────────────────────────────────
df = pd.DataFrame({
    "order_id": [f"ORD-{i+1:05d}" for i in range(N)],
    "order_date": order_dates,
    "year": pd.DatetimeIndex(order_dates).year,
    "quarter": ["Q" + str(q) for q in pd.DatetimeIndex(order_dates).quarter],
    "month": pd.DatetimeIndex(order_dates).month,
    "month_name": pd.DatetimeIndex(order_dates).strftime("%B"),
    "region": regions,
    "country": countries,
    "category": categories,
    "subcategory": subcategories,
    "customer_segment": customer_segments,
    "quantity": quantities,
    "unit_price": unit_prices,
    "revenue": revenues,
    "cost": total_costs,
    "profit": profits,
    "profit_margin": profit_margins,
})

df = df.sort_values("order_date").reset_index(drop=True)

out_path = os.path.join(os.path.dirname(__file__), "data", "global_retail_sales.csv")
df.to_csv(out_path, index=False)

# ── Summary ─────────────────────────────────────────────────────
print(f"✅ Dataset generated: {len(df):,} rows → {out_path}")
print(f"   Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
print(f"   Regions: {df['region'].nunique()} | Countries: {df['country'].nunique()}")
print(f"   Categories: {df['category'].nunique()} | Subcategories: {df['subcategory'].nunique()}")
print()
for yr in sorted(df['year'].unique()):
    sub = df[df['year'] == yr]
    print(f"   {yr}: {len(sub):,} transactions | Revenue: ${sub['revenue'].sum():,.2f} | Profit: ${sub['profit'].sum():,.2f}")
print(f"\n   Total Revenue: ${df['revenue'].sum():,.2f}")
print(f"   Total Profit:  ${df['profit'].sum():,.2f}")
