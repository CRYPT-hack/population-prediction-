#!/usr/bin/env python3
"""
Extract the few rows the model needs from the official Census 2011 spreadsheets.

Raw files (not committed, ~9 MB; download from censusindia.gov.in or ask the group):
    data/raw/DS-0600-D02-MDDS.XLSX   D-02 migration table, Haryana
    data/raw/DS-0700-D02-MDDS.XLSX   D-02 migration table, NCT of Delhi
    data/raw/DS-0900-D02-MDDS.XLSX   D-02 migration table, Uttar Pradesh
    data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx   Primary Census Abstract 2011 (state + district totals)

D-02 = "Migrants classified by place of last residence, sex and duration of residence in the place of
enumeration". Migrants who have lived in the district for LESS THAN 1 YEAR are, to a good approximation,
the people who moved there during the year before the census, i.e. an annual flow.

Output (committed, small):  data/census_d02_flows.csv, data/census_population_2011.csv
Run:  python data/extract_census.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

COLS = ["table", "state", "district", "area", "last", "last_tru", "enum_tru",
        "tot", "tot_m", "tot_f", "lt1", "lt1_m", "lt1_f", "y1_4", "y1_4_m", "y1_4_f",
        "y5_9", "y5_9_m", "y5_9_f", "y10_19", "y10_19_m", "y10_19_f", "y20", "y20_m", "y20_f",
        "ns", "ns_m", "ns_f"]
ORIGINS = ["Total", "Elsewhere in the district of enumeration", "In other districts of the state of enumeration",
           "States in India beyond the state of enumeration", "Haryana", "NCT of Delhi", "Uttar Pradesh",
           "Last residence outside India"]
AREAS = {  # file -> {area name in file: our label}
    "DS-0700-D02-MDDS.XLSX": {"State - NCT OF DELHI": "Delhi"},
    "DS-0600-D02-MDDS.XLSX": {"Gurgaon": "Gurugram", "Faridabad": "Faridabad"},
    "DS-0900-D02-MDDS.XLSX": {"Gautam Buddha Nagar": "Noida"},
}
PCA = {"NCT OF DELHI": "Delhi", "Gurgaon": "Gurugram", "Gautam Buddha Nagar": "Noida", "Faridabad": "Faridabad",
       "HARYANA": "Haryana (state)", "UTTAR PRADESH": "Uttar Pradesh (state)"}


def extract_flows() -> pd.DataFrame:
    rows = []
    for fname, areas in AREAS.items():
        df = pd.read_excel(RAW / fname, header=None, skiprows=5, names=COLS)
        sel = df[df.area.isin(areas) & (df.last_tru == "Total") & (df.enum_tru == "Total") & df["last"].isin(ORIGINS)]
        for _, r in sel.iterrows():
            rows.append({"destination": areas[r.area], "last_residence": r["last"],
                         "total_migrants": int(r.tot), "moved_last_year": int(r.lt1), "moved_1_4_years": int(r.y1_4)})
    return pd.DataFrame(rows)


def extract_population() -> pd.DataFrame:
    df = pd.read_excel(RAW / "DDW_PCA0000_2011_Indiastatedist.xlsx")
    df = df[(df.TRU == "Total") & df.Name.isin(PCA) & df.Level.isin(["STATE", "DISTRICT"])]
    return pd.DataFrame({"region": df.Name.map(PCA), "population_2011": df.TOT_P.astype(int)})


if __name__ == "__main__":
    f = extract_flows(); p = extract_population()
    f.to_csv(HERE / "census_d02_flows.csv", index=False)
    p.to_csv(HERE / "census_population_2011.csv", index=False)
    print(f.to_string(index=False)); print(p.to_string(index=False))
