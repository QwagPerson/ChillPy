# chillPy

`chillPy` is an early pure Python rewrite of the R package `chillR`, a toolkit
for phenology analysis, winter chill, heat accumulation, weather processing,
and climate scenario workflows for temperate fruit trees.

The original R source is included in this repository under `chillR - RPackage/`
and is currently used as the API reference for this rewrite.

## Status

This repository is in an incremental porting phase. The Python package
structure, public API, tests, and API mapping document are present, and the
lower-level temperature models plus the first hourly-temperature and chill
accumulation utilities have been translated from the R source.

Current implementation policy:

- keep the Python package importable;
- expose Pythonic `snake_case` names;
- preserve R-style aliases where they are valid Python identifiers;
- prefer translated scientific behavior over placeholder values;
- keep remaining placeholders explicit and documented in the API mapping;
- raise `NotImplementedError` where behavior should not be mocked silently.

## Example

```python
import pandas as pd
from chillPy import chilling, stack_hourly_temps

daily_weather = pd.DataFrame(
    {
        "Year": [2020, 2021],
        "Month": [12, 1],
        "Day": [31, 1],
        "JDay": [366, 1],
        "Tmin": [5.0, 5.0],
        "Tmax": [5.0, 5.0],
    }
)

hourly = stack_hourly_temps(daily_weather, latitude=0)
summary = chilling(hourly, start_jday=366, end_jday=1)
```

Daily and hourly records can also be completed before analysis:

```python
from chillPy import add_date, make_JDay, make_all_day_table

daily_weather = make_JDay(daily_weather)
daily_weather = add_date(daily_weather)
complete_daily = make_all_day_table(daily_weather, timestep="day")
```

Auxiliary daily records can be used to patch missing daily extremes with
interval-specific bias correction:

```python
from chillPy import patch_daily_temps

patched = patch_daily_temps(daily_weather, {"nearby_station": daily_weather})
```

Hourly records with gaps can be filled with the translated Linvill-guided
interpolation workflow:

```python
from chillPy import interpolate_gaps_hourly

filled_hourly = interpolate_gaps_hourly(hourly["hourtemps"], latitude=0)["weather"]
```

Daily weather records can be completed, interpolated, and checked before
downstream chill or heat calculations:

```python
from chillPy import check_temperature_record, fix_weather

fixed = fix_weather(daily_weather, start_year=2020, end_year=2021)
check = check_temperature_record(fixed["weather"])
```

Downloaded daily weather tables can be normalized before cleaning:

```python
from chillPy import weather2chillR

weather = weather2chillR(downloaded_weather, database="CIMIS")
```

Phenology seasons and observed stage transitions can be prepared for later
modeling:

```python
from chillPy import genSeasonList, stage_transitions

season_tables = genSeasonList(hourly["hourtemps"], years=[2021])
transitions = stage_transitions(observations, hourly, stages=["green_tip", "bloom"])
```

Non-plotting PLS phenology summaries are available for prepared daily records:

```python
from chillPy import PLS_pheno

pls = PLS_pheno(daily_weather, bloom_dates, split_month=6, use_Tmean=False)
summary = pls["PLS_summary"]
```

PLS outputs can be converted into interpretation tables without rendering:

```python
from chillPy import prepare_pls_plot_data

plot_data = prepare_pls_plot_data(pls, vip_threshold=0.8)
important_windows = plot_data["important_windows"]
```

PLS and daily chill summaries can also be rendered with matplotlib. Library
functions return figure and axes objects and never call `show()`:

```python
from chillPy import make_chill_plot, plot_pls

chill_plot = make_chill_plot({"daily_chill": daily_metrics}, metrics=["Chill_Portions"])
pls_plot = plot_pls(pls, vip_threshold=0.8)
```

Bloom dates can be predicted from sequential chill and heat requirements:

```python
from chillPy import bloom_prediction3

predictions = bloom_prediction3(
    hourly["hourtemps"],
    chill_req=[40, 50],
    heat_req=[1200, 1500],
)
```

Deterministic phenology model wrappers are available for fitting workflows:

```python
from chillPy import UniChill_Wrapper

season = hourly["hourtemps"][["Temp", "JDay"]]
predicted_jday = UniChill_Wrapper(season, [0, 0, 0, 0, 0, 40, 1200])
```

Phenology wrapper parameters can be fitted against observed bloom dates:

```python
import pandas as pd
from chillPy import phenologyFitter

season = pd.DataFrame({"Temp": [5, 5, 5, 5, 5], "JDay": [350, 351, 1, 2, 3]})
fit = phenologyFitter(
    par_guess=[0, 0, 2, 1],
    bloom_jdays=[2],
    season_list=[season],
    modelfn="UniForce_Wrapper",
)
residuals = fit["residuals"]
```

## Examples / Notebooks

The `examples/` directory contains reproducible getting-started notebooks that
double as smoke tests for complete workflows. They use deterministic synthetic
data in `examples/data/`; these CSV files are not field observations.

Recommended order:

1. [`01_getting_started_weather_and_chill.ipynb`](examples/01_getting_started_weather_and_chill.ipynb)
   covers weather checking, gap repair, hourly temperature generation, chill
   metrics, heat accumulation, and daily chill plots.
2. [`02_temperature_interpolation_and_gap_filling.ipynb`](examples/02_temperature_interpolation_and_gap_filling.ipynb)
   demonstrates one-dimensional and hourly temperature gap filling with before
   and after plots.
3. [`03_phenology_and_bloom_prediction.ipynb`](examples/03_phenology_and_bloom_prediction.ipynb)
   runs a simple bloom prediction workflow and compares predicted and synthetic
   observed bloom dates.
4. [`04_pls_analysis.ipynb`](examples/04_pls_analysis.ipynb)
   runs a compact PLS phenology analysis, extracts interpretation tables, and
   renders the matplotlib PLS summary plot.
5. [`05_train_phenoflex_cherry_bloom_prediction.ipynb`](examples/05_train_phenoflex_cherry_bloom_prediction.ipynb)
   fits and compares PhenoFlex-style cherry bloom prediction models using
   synthetic hourly weather and bloom observations.

Run them locally after installing the development dependencies:

```bash
pip install -e ".[dev]"
jupyter notebook examples/
```

The notebooks and example data can be regenerated with:

```bash
python scripts/create_example_notebooks.py
```

Automated notebook execution is included in the pytest suite and can also be
run directly:

```bash
pytest tests/test_notebooks.py
```

## Installation

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

The tests cover implemented numerical and plotting behavior and keep remaining
external-service placeholders importable and explicit.

## API Mapping

See [`docs/api_mapping.md`](docs/api_mapping.md) for the mapping from exported R
functions to proposed Python functions and modules.
