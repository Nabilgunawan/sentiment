import csv
import io
import json
from datetime import datetime


def export_to_csv(df):
    output = io.StringIO()
    writer = csv.writer(output)
    export_cols = [c for c in df.columns if c not in ("clean_text",)]
    writer.writerow(export_cols)
    for _, row in df.iterrows():
        writer.writerow([
            "" if pd.isna(row.get(c)) else str(row[c])
            if not isinstance(row.get(c), (int, float))
            else row[c]
            for c in export_cols
        ])
    return output.getvalue()


def export_to_json(df):
    export_cols = [c for c in df.columns if c not in ("clean_text",)]
    rows = []
    for _, row in df.iterrows():
        r = {}
        for c in export_cols:
            val = row.get(c)
            if isinstance(val, datetime):
                val = val.isoformat()
            elif pd.isna(val):
                val = None
            elif isinstance(val, (int, float)):
                val = val
            else:
                val = str(val)
            r[c] = val
        rows.append(r)
    return json.dumps(rows, indent=2, default=str)


def export_summary_text(summary_text):
    return summary_text


import pandas as pd
