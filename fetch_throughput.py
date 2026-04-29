"""
Fetch hourly picker & packer throughput per CH for the last 28 days.
Saves merged CSV to data/exports/picker_packer_hourly_throughput_4w.csv

Run:
    cd /Users/siddansh/Code && python3 src/scripts/fetch_picker_packer_throughput.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path("/Users/siddansh/Code")
sys.path.insert(0, str(REPO_ROOT / "reporting_langgraph"))

from src.clients.superset import SupersetClient  # noqa: E402
from src.settings import settings  # noqa: E402

EXPORT_DIR = REPO_ROOT / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

DB_SUPPLY = 2

# Source-of-truth orders/units at fo.created_at grain. Aligns with the Ops
# Rolling Hourly Orders CSV which is built from the same definition.
CREATED_SQL = """
SELECT DATE(fo.created_at)                      AS event_date,
       HOUR(fo.created_at)                      AS event_hour,
       fo.warehouse_id                          AS ch_id,
       COUNT(DISTINCT fo.id)                    AS orders_created,
       SUM(COALESCE(foi.requested_quantity, 0)) AS units_requested
FROM inventory.fulfilment_orders fo
INNER JOIN inventory.fulfilment_order_item foi ON foi.fulfilment_order_id = fo.id
WHERE fo.created_at >= DATE_SUB(CURDATE(), INTERVAL 28 DAY)
  AND fo.warehouse_id LIKE 'FC%'
  AND fo.id NOT LIKE 'SB%'
  AND fo.id NOT LIKE 'SL%'
  AND fo.processing_status NOT IN (
      'CANCELLED','ALLOCATION_FAILED','PACKING_FAILED','HANDOVER_FAILED'
  )
GROUP BY DATE(fo.created_at), HOUR(fo.created_at), fo.warehouse_id
ORDER BY event_date DESC, event_hour, ch_id
"""

PICKER_SQL = """
SELECT DATE(a.created_at) AS event_date,
       HOUR(a.created_at) AS event_hour,
       fo.warehouse_id    AS ch_id,
       COUNT(DISTINCT a.user_id)              AS active_pickers,
       COUNT(DISTINCT a.fulfilment_order_id)  AS orders_picked,
       ROUND(SUM(foi_units.units), 0)         AS units_picked
FROM inventory.fulfilment_order_audits a
JOIN inventory.fulfilment_orders fo ON fo.id = a.fulfilment_order_id
JOIN (
    SELECT fulfilment_order_id, SUM(picked_quantity) AS units
    FROM inventory.fulfilment_order_item
    WHERE picked_quantity > 0
    GROUP BY fulfilment_order_id
) foi_units ON foi_units.fulfilment_order_id = a.fulfilment_order_id
WHERE a.to_status = 'PICKED'
  AND a.created_at >= DATE_SUB(CURDATE(), INTERVAL 28 DAY)
  AND fo.warehouse_id LIKE 'FC%'
  AND fo.id NOT LIKE 'SB%'
  AND fo.id NOT LIKE 'SL%'
GROUP BY DATE(a.created_at), HOUR(a.created_at), fo.warehouse_id
ORDER BY event_date DESC, event_hour, ch_id
"""

PACKER_SQL = """
SELECT DATE(a.created_at) AS event_date,
       HOUR(a.created_at) AS event_hour,
       fo.warehouse_id    AS ch_id,
       COUNT(DISTINCT a.user_id)              AS active_packers,
       COUNT(DISTINCT a.fulfilment_order_id)  AS orders_packed,
       ROUND(SUM(foi_units.units), 0)         AS units_packed
FROM inventory.fulfilment_order_audits a
JOIN inventory.fulfilment_orders fo ON fo.id = a.fulfilment_order_id
JOIN (
    SELECT fulfilment_order_id, SUM(packed_quantity) AS units
    FROM inventory.fulfilment_order_item
    WHERE packed_quantity > 0
    GROUP BY fulfilment_order_id
) foi_units ON foi_units.fulfilment_order_id = a.fulfilment_order_id
WHERE a.to_status = 'PACKED'
  AND a.created_at >= DATE_SUB(CURDATE(), INTERVAL 28 DAY)
  AND fo.warehouse_id LIKE 'FC%'
  AND fo.id NOT LIKE 'SB%'
  AND fo.id NOT LIKE 'SL%'
GROUP BY DATE(a.created_at), HOUR(a.created_at), fo.warehouse_id
ORDER BY event_date DESC, event_hour, ch_id
"""


def fetch(client: SupersetClient, auth, sql: str) -> pd.DataFrame:
    resp = client.execute_sql(auth, database_id=DB_SUPPLY, sql=sql, limit=200_000)
    cols, rows = SupersetClient.extract_rows(resp)
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    print("Authenticating Superset...")
    client = SupersetClient(
        base_url=settings.superset_base_url,
        username=settings.superset_username,
        password=settings.superset_password,
        provider=settings.superset_provider,
        timeout_seconds=180,
    )
    auth = client.authenticate()

    print("Fetching created-at hourly orders/units (4 weeks)...")
    created = fetch(client, auth, CREATED_SQL)
    print(f"  rows: {len(created)}")

    print("Fetching picker hourly throughput (4 weeks)...")
    pickers = fetch(client, auth, PICKER_SQL)
    print(f"  rows: {len(pickers)}")

    print("Fetching packer hourly throughput (4 weeks)...")
    packers = fetch(client, auth, PACKER_SQL)
    print(f"  rows: {len(packers)}")

    merged = pd.merge(
        pickers, packers,
        on=["event_date", "event_hour", "ch_id"],
        how="outer",
    )
    merged = pd.merge(
        created, merged,
        on=["event_date", "event_hour", "ch_id"],
        how="outer",
    ).fillna(0)

    for col in ["orders_created", "units_requested",
                "active_pickers", "orders_picked", "units_picked",
                "active_packers", "orders_packed", "units_packed",
                "event_hour"]:
        merged[col] = merged[col].astype(int)

    merged["units_per_picker_per_hour"] = (
        merged["units_picked"] / merged["active_pickers"].replace(0, pd.NA)
    ).round(1)
    merged["units_per_packer_per_hour"] = (
        merged["units_packed"] / merged["active_packers"].replace(0, pd.NA)
    ).round(1)

    merged = merged.sort_values(
        ["event_date", "event_hour", "ch_id"],
        ascending=[False, True, True],
    )

    out = EXPORT_DIR / "picker_packer_hourly_throughput_4w.csv"
    merged.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    print(f"Rows: {len(merged):,}  CHs: {merged['ch_id'].nunique()}  "
          f"Date range: {merged['event_date'].min()} → {merged['event_date'].max()}")
    print("\nNetwork averages (period):")
    print(f"  Picker UPH: {merged['units_per_picker_per_hour'].mean():.1f}")
    print(f"  Packer UPH: {merged['units_per_packer_per_hour'].mean():.1f}")
    print("\nPer-CH avg UPH:")
    summary = merged.groupby("ch_id").agg(
        picker_uph=("units_per_picker_per_hour", "mean"),
        packer_uph=("units_per_packer_per_hour", "mean"),
        hours_observed=("event_date", "count"),
    ).round(1).sort_values("picker_uph", ascending=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
