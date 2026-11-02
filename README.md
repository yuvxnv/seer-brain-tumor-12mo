# Predicting 12-month death from brain tumors (SEER)

A learning project in ML for oncology research. Using SEER data on malignant brain tumors diagnosed 2000-2023,
it predicts whether a patient **dies of this cancer within 12 months of diagnosis**.

## Data

SEER Research Data from 17 registries, diagnosis years 2000-2023, malignant brain tumors only (123,487 records before cleaning),
exported through SEER*Stat. Access is free but must be requested: https://seer.cancer.gov/data/access.html

**No data is included in this repository.** SEER's data use agreement forbids redistributing the data, so you need your own approved copy.
See [data/README.md](data/README.md) for the columns I exported.

## Setup

```bash
pip install -r requirements.txt
# get your own data, see data/README.md (SEER data cannot be redistributed)
python explore.py
```

## Method

- **Target:** 1 if died of this cancer with survival <= 12 months; 0 if survival > 12 months.
  Patients alive or dead of other causes within 12 months are dropped (outcome unknown). After cleaning: 109,956 rows, 46.1% positive.
- **Leakage rule:** `months`, `vital` and `cod` build the target and are never used as features.
- **Features:** age group, sex, year, histology code, stage, grade, radiation, chemo.
- **Models:** logistic regression (interpretable) and random forest.

## Results

| Experiment | AUC |
|---|---|
| Logistic regression, one 80/20 split | 0.875 |
| Without radiation | 0.872 |
| Without chemo | 0.868 |
| Without radiation and chemo | 0.852 |
| Logistic regression, 5-fold CV | 0.871 (std 0.003) |
| Random forest, 5-fold CV | 0.876 (std 0.003) |

The forest beat logistic regression in all 5 folds, but only by about 0.005 AUC. Logistic regression is preferred here for its readable coefficients.

## Limitations

- **Immortal time bias:** radiation and chemo can only happen to patients who live long enough to receive them, so these columns partly encode survival time.
  Their coefficients (e.g. chemo = Yes, -0.64) show association, **not** a treatment effect.
- The ablation shows the model does not mainly depend on treatment, but cannot prove the columns are uncontaminated.
- Random forest was not tuned. No calibration check yet. Histology coefficients for rare codes may rest on few patients.
- **Surgery was excluded.** The surgery field in my export (`RX Summ--Surg Prim Site 2023+`) is only recorded for 2023 diagnoses, so it is about 96% blank.
- **Malignant tumors only.** Results do not apply to benign or borderline tumors.
- **Grade comes from two systems, merged with an assumed mapping.** Diagnoses through 2017 use a differentiation-based grade; 2018 and later use SEER's site-specific grade categories.
  I mapped both onto Grade 1-4 myself, so the same label may not mean the same thing across eras. About half of the final rows (49%) have grade "Unknown".
- **Chemotherapy is known to be underreported in SEER,** and "No/Unknown" mixes untreated patients with unrecorded ones.
- **The 12-month framing loses patients and ignores follow-up time.** Dropping patients without a clear 12-month outcome removes about 8% of cleaned rows
  (11% of the raw export once rows with missing survival time or unknown cause of death are included). It treats the outcome as a yes/no at 12 months,
  so it does not handle censoring the way survival analysis does.
- **No external validation.** All results come from the same SEER export.
- **Not for clinical use.** This is an educational model built on population-level registry data.

## How it was built

I used Claude Code for implementation and made the analysis decisions myself.
