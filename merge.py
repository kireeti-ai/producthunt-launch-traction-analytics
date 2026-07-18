import os
import json
import pandas as pd

DATA_FOLDER = "producthunt_data"

all_rows = []

for file in sorted(os.listdir(DATA_FOLDER)):

    if not file.endswith(".json"):
        continue

    path = os.path.join(DATA_FOLDER, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

        if isinstance(data, list):
            all_rows.extend(data)

print(f"Rows before cleaning : {len(all_rows)}")

df = pd.DataFrame(all_rows)

print(df.head())

df.drop_duplicates(
    subset=["period", "url"],
    inplace=True
)

print(f"Rows after removing duplicates : {len(df)}")

df = df.sort_values(
    ["period", "rank"]
)

df.to_csv(
    "producthunt_master.csv",
    index=False
)

print("Saved producthunt_master.csv")
