# AI-Assisted Optimization of Industrial Rotating Equipment — code and data

[![tests](https://github.com/demiral10akbar/MEASUR_ai/actions/workflows/tests.yml/badge.svg)](https://github.com/demiral10akbar/MEASUR_ai/actions/workflows/tests.yml)

Reproducible calculations and reference AI modules for the manuscript

> **AI-Assisted Optimization of Industrial Rotating Equipment: A Unified Energy Efficiency Framework for Fans, Pumps, Motors, and Compressors**

The study combines physics-based energy modelling with the U.S. DOE MEASUR toolbox
and data-driven AI methods for five SME case studies (compressed air, fans, pumps,
and a motor-driven mixer). This repository lets readers:

1. **recompute every KPI** in Tables 4–9 (energy, cost and CO₂ savings, payback) from the
   MEASUR baseline and scenario results, using Eqs. (1)–(5) of the manuscript;
2. **reproduce the sensitivity analyses** of Section 3.5 (grid emission factor,
   electricity price, implementation cost);
3. **inspect and run the reference AI pipelines** specified in Table A1 (model type,
   inputs, splits, optimizer, set-point logic);
4. **plug in field measurements** through schema-checked CSV templates.

## Scope — please read

* **MEASUR is not executed by this code.** MEASUR is a separate DOE application. The
  baseline and scenario energy and peak-demand values in `data/facilities.yaml` are the
  MEASUR outputs reported in the manuscript; this code applies the KPI equations to them
  and checks the published numbers.
* **No AI model in this repository was trained on facility data.** As stated in
  Section 3.3.6 of the manuscript, the study is a scenario analysis of AI-enabled
  operation: the AI effect enters MEASUR as assumed parameter changes (Table A2). The
  modules in `src/rotaeq/ai/` implement the configuration recommended in Table A1 for
  deployment and field validation.
* `scripts/demo_ai_modules.py` runs those pipelines on **synthetic data** only, to show
  they work end to end. Its metrics are **not** results for the case-study facilities.
* Every value carries the manuscript's data-source flag: **M** measured, **S** simulated
  with MEASUR, **E** estimated cost.

## Repository layout

```
data/
  facilities.yaml        baseline and scenario results per facility (Tables 3-8, A2)
  assumptions.yaml       prices, demand charges, emission factors (Section 3.5)
  field/                 place field measurements here (schemas in field/README.md)
    templates/           empty CSV templates with the required columns
src/rotaeq/
  kpi.py                 Eqs. (1)-(5): energy saving, CO2, cost, payback, affinity law
  physics.py             leakage/airflow, affinity-law speed ratios, mixer Reynolds number
  analysis.py            recomputation of Tables 4-9 and consistency checks
  sensitivity.py         emission-factor, price and cost sensitivity (Section 3.5)
  io.py                  schema-checked field-data loading, leakage from load/unload logs
  ai/                    reference AI modules of Table A1
    splits.py            chronological and group-wise 70/15/15 splits
    features.py          vibration features (RMS, 1x/2x running speed, broadband)
    models.py            MLP builders (Adam) and early stopping on explicit validation sets
    setpoints.py         pressure, fan-speed and mixer-speed set-point translation
    synthetic.py         SYNTHETIC demo data generators
scripts/
  reproduce_tables.py    recompute Tables 4-9 and compare with the manuscript
  run_sensitivity.py     sensitivity analyses of Section 3.5
  make_figures.py        cross-system comparison figures
  demo_ai_modules.py     run the AI pipelines on synthetic data
tests/                   pytest suite (KPI equations, reproduction, AI utilities, I/O)
results/                 outputs written by the scripts
```

## Installation

```bash
git clone https://github.com/demiral10akbar/MEASUR_ai.git
cd MEASUR_ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Python ≥ 3.9. Dependencies: NumPy, pandas, scikit-learn, Matplotlib, PyYAML.

## Usage

```bash
python scripts/reproduce_tables.py    # Tables 4-9; exits with code 1 if any value disagrees
python scripts/run_sensitivity.py     # Section 3.5 sensitivity analyses
python scripts/make_figures.py        # results/fig_*.png
python scripts/demo_ai_modules.py     # AI pipelines on SYNTHETIC data
pytest -q                             # test suite
```

## Mapping to the manuscript

| Manuscript | Code |
|---|---|
| Eqs. (1)–(5), Section 3.4 | `rotaeq.kpi` |
| Tables 4–8 (facility results) | `data/facilities.yaml`, `rotaeq.analysis.scenario_kpis` |
| Table 9 (cross-system summary) | `rotaeq.analysis.cross_system_summary` |
| Section 3.5 (economic and environmental assumptions) | `data/assumptions.yaml`, `rotaeq.sensitivity` |
| Table A1 (AI module specification) | `rotaeq.ai` |
| Table A2 (assumed parameter changes) | `assumptions:` blocks in `data/facilities.yaml` |
| Figures 9–10 | `scripts/make_figures.py` |

## Key assumptions

* Electricity prices (USD/kWh, 2025): A 0.100, B 0.102, E 0.120 from the facilities' actual
  electricity bills, confirmed in site interviews; C and D 0.066 (MEASUR default). Demand charges
  USD 60/kW·yr (A) and 120/kW·yr (B).
* Grid emission factor: 401.07 kg CO₂/MWh (MEASUR default, U.S. average eGRID). The
  official 2022 Turkish consumption-point factors (442 and 474 kg CO₂/MWh; ETKB, 2024)
  are included for comparison and scale all CO₂ reductions by ≈1.10 and ≈1.18.
* Implementation costs are the authors' engineering estimates (flag E).

## Adding field data

See `data/field/README.md`. Load a file with

```python
from rotaeq.io import load_field_csv, leakage_fraction_from_log
log = load_field_csv("data/field/compressor_log.csv")
print(leakage_fraction_from_log(log[log["timestamp"].dt.hour < 5]))   # non-production hours
```

Replace the synthetic generators in `scripts/demo_ai_modules.py` with these loaders to
train and validate the AI modules on facility data, and update the `M`-flagged inputs in
`data/facilities.yaml`.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`).

## License

MIT — see `LICENSE`.
