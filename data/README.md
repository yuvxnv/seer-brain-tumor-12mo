# Data

`export.csv` is **not included**. SEER data cannot be redistributed, so you need your own copy.

1. Request access to SEER Research Data and open SEER*Stat (https://seer.cancer.gov/data/access.html).
2. Build a case listing session in SEER Research Data: brain tumors, malignant behavior only, diagnosed 2000-2023 (the original export had 123,487 rows).
3. Export these columns as CSV: Age recode with <1 year olds and 90+, Sex, Year of diagnosis,
   Behavior code ICD-O-3, Histologic Type ICD-O-3, Combined Summary Stage with Expanded Regional Codes (2004+),
   RX Summ--Surg Prim Site 2023 (2023+), Radiation recode, Chemotherapy recode (yes, no/unk),
   Survival months, Vital status recode (study cutoff used), SEER cause-specific death classification,
   Derived Summary Grade 2018 (2018+), Grade Recode (thru 2017).
4. Save as `data/export.csv`.
