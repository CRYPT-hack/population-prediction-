# Data sources

| File | Content | Source |
|---|---|---|
| `census_d02_flows.csv` | Migrants by place of last residence and duration of residence, for Delhi (state), Gurgaon, Faridabad and Gautam Buddha Nagar | Census of India 2011, Table D-02 (DS-0600/0700/0900-D02-MDDS.XLSX), extracted by `extract_census.py` |
| `census_population_2011.csv` | 2011 population of the four regions and of Haryana and Uttar Pradesh | Census of India 2011, Primary Census Abstract (DDW_PCA0000_2011_Indiastatedist.xlsx) |
| `srs_2001_2011.csv` | Crude birth and death rates per 1000, 2001-2011, for Delhi, Haryana, Uttar Pradesh | Sample Registration System bulletins, as tabulated in RBI Handbook of Statistics on Indian States, Tables 8-9 |
| `delhi_migration_2001_2012.csv` | Delhi births, deaths, natural increase and net migration (lakh) | Economic Survey of Delhi 2014-15, Chapter 2, Statements 2.16-2.17 (Directorate of Economics & Statistics, GNCTD; CRS-based) |

The raw Census spreadsheets (about 9 MB) are not committed; put them in `data/raw/` and run
`python data/extract_census.py` to regenerate the two census CSVs.
