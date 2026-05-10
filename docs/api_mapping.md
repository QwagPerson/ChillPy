# chillR to chillPy API Mapping

This document maps exported functions from the original R `chillR` package to
the initial Python scaffold. Status values:

- `stub`: placeholder exists, scientific logic not ported.
- `partial`: minimal non-scientific behavior exists for convenience.
- `implemented`: translated from the R implementation with focused tests.
- `not started`: no Python placeholder yet.

| R function | Python function | Module | Status |
|---|---|---|---|
| `Chilling_Hours` | `chilling_hours` | `chillPy.temperature_models` | implemented |
| `Date2YEARMODA` | `date_to_yearmoda` | `chillPy.date_utils` | implemented |
| `DynModel_driver` | `dynamic_model_driver` | `chillPy.temperature_models` | implemented |
| `Dynamic_Model` | `dynamic_model` | `chillPy.temperature_models` | implemented |
| `Empirical_daily_temperature_curve` | `empirical_daily_temperature_curve` | `chillPy.temperature` | stub |
| `Empirical_hourly_temperatures` | `empirical_hourly_temperatures` | `chillPy.temperature` | stub |
| `GDD` | `gdd` | `chillPy.temperature_models` | implemented |
| `GDH` | `gdh` | `chillPy.temperature_models` | implemented |
| `GDH_model` | `gdh_model` | `chillPy.temperature_models` | implemented |
| `JDay_count` | `jday_count` | `chillPy.date_utils` | implemented |
| `JDay_earlier` | `jday_earlier` | `chillPy.date_utils` | implemented |
| `JDay_later` | `jday_later` | `chillPy.date_utils` | implemented |
| `PLS_chill_force` | `pls_chill_force` | `chillPy.phenology` | implemented |
| `PLS_pheno` | `pls_pheno` | `chillPy.phenology` | implemented |
| `PhenoFlex` | `phenoflex` | `chillPy.temperature_models` | stub |
| `PhenoFlex_GAUSSwrapper` | `phenoflex_gauss_wrapper` | `chillPy.phenology` | stub |
| `PhenoFlex_GDHwrapper` | `phenoflex_gdh_wrapper` | `chillPy.phenology` | stub |
| `PhenoFlex_fixedDynModelGAUSSwrapper` | `phenoflex_fixed_dynamic_model_gauss_wrapper` | `chillPy.phenology` | stub |
| `PhenoFlex_fixedDynModelwrapper` | `phenoflex_fixed_dynamic_model_wrapper` | `chillPy.phenology` | stub |
| `RMSEP` | `rmsep` | `chillPy.metrics` | implemented |
| `RPD` | `rpd` | `chillPy.metrics` | implemented |
| `RPIQ` | `rpiq` | `chillPy.metrics` | implemented |
| `StepChill_Wrapper` | `step_chill_wrapper` | `chillPy.phenology` | implemented |
| `UniChill_Wrapper` | `uni_chill_wrapper` | `chillPy.phenology` | implemented |
| `UniForce_Wrapper` | `uni_force_wrapper` | `chillPy.phenology` | implemented |
| `UnifiedModel_Wrapper` | `unified_model_wrapper` | `chillPy.phenology` | implemented |
| `Utah_Model` | `utah_model` | `chillPy.temperature_models` | implemented |
| `VIP` | `vip` | `chillPy.phenology` | implemented |
| `YEARMODA2Date` | `yearmoda_to_date` | `chillPy.date_utils` | implemented |
| `add_date` | `add_date` | `chillPy.date_utils` | implemented |
| `bloom_prediction` | `bloom_prediction` | `chillPy.phenology` | implemented |
| `bloom_prediction2` | `bloom_prediction2` | `chillPy.phenology` | implemented |
| `bloom_prediction3` | `bloom_prediction3` | `chillPy.phenology` | implemented |
| `bootstrap.phenologyFit` | `bootstrap_phenology_fit` | `chillPy.phenology` | stub |
| `check_temperature_record` | `check_temperature_record` | `chillPy.weather` | implemented |
| `check_temperature_scenario` | `check_temperature_scenario` | `chillPy.weather` | stub |
| `chile_agromet2chillR` | `chile_agromet_to_chillr` | `chillPy.weather` | implemented |
| `chilling` | `chilling` | `chillPy.temperature` | implemented |
| `chilling_hourtable` | `chilling_hourtable` | `chillPy.temperature` | implemented |
| `color_bar_maker` | `color_bar_maker` | `chillPy.plotting` | implemented |
| `convert_scen_information` | `convert_scen_information` | `chillPy.scenarios` | stub |
| `daily_chill` | `daily_chill` | `chillPy.temperature` | implemented |
| `daylength` | `daylength` | `chillPy.date_utils` | implemented |
| `download_baseline_cmip6_ecmwfr` | `download_baseline_cmip6_ecmwfr` | `chillPy.scenarios` | stub |
| `download_cmip6_ecmwfr` | `download_cmip6_ecmwfr` | `chillPy.scenarios` | stub |
| `extract_cmip6_data` | `extract_cmip6_data` | `chillPy.scenarios` | stub |
| `extract_differences_between_characters` | `extract_differences_between_characters` | `chillPy.utils` | stub |
| `extract_temperatures_from_grids` | `extract_temperatures_from_grids` | `chillPy.scenarios` | stub |
| `filter_temperatures` | `filter_temperatures` | `chillPy.temperature` | implemented |
| `fix_weather` | `fix_weather` | `chillPy.weather` | implemented |
| `genSeason` | `gen_season` | `chillPy.phenology` | implemented |
| `genSeasonList` | `gen_season_list` | `chillPy.phenology` | implemented |
| `gen_rel_change_scenario` | `gen_rel_change_scenario` | `chillPy.scenarios` | stub |
| `getClimateWizardData` | `get_climate_wizard_data` | `chillPy.scenarios` | stub |
| `getClimateWizard_scenarios` | `get_climate_wizard_scenarios` | `chillPy.scenarios` | stub |
| `get_last_date` | `get_last_date` | `chillPy.date_utils` | implemented |
| `get_weather` | `get_weather` | `chillPy.weather` | stub |
| `handle_cimis` | `handle_cimis` | `chillPy.weather` | stub |
| `handle_dwd` | `handle_dwd` | `chillPy.weather` | stub |
| `handle_dwd_old` | `handle_dwd_old` | `chillPy.weather` | stub |
| `handle_gsod` | `handle_gsod` | `chillPy.weather` | stub |
| `handle_gsod_old` | `handle_gsod_old` | `chillPy.weather` | stub |
| `handle_ucipm` | `handle_ucipm` | `chillPy.weather` | stub |
| `identify_common_string` | `identify_common_string` | `chillPy.utils` | partial |
| `interpolate_gaps` | `interpolate_gaps` | `chillPy.temperature` | implemented |
| `interpolate_gaps_hourly` | `interpolate_gaps_hourly` | `chillPy.temperature` | implemented |
| `leap_year` | `leap_year` | `chillPy.date_utils` | implemented |
| `load_ClimateWizard_scenarios` | `load_climate_wizard_scenarios` | `chillPy.scenarios` | stub |
| `load_temperature_scenarios` | `load_temperature_scenarios` | `chillPy.scenarios` | stub |
| `make_JDay` | `make_jday` | `chillPy.date_utils` | implemented |
| `make_all_day_table` | `make_all_day_table` | `chillPy.temperature` | implemented |
| `make_california_UCIPM_station_list` | `make_california_ucipm_station_list` | `chillPy.weather` | stub |
| `make_chill_plot` | `make_chill_plot` | `chillPy.plotting` | stub |
| `make_climate_scenario` | `make_climate_scenario` | `chillPy.scenarios` | stub |
| `make_climate_scenario_from_files` | `make_climate_scenario_from_files` | `chillPy.scenarios` | stub |
| `make_daily_chill_figures` | `make_daily_chill_figures` | `chillPy.plotting` | stub |
| `make_daily_chill_plot` | `make_daily_chill_plot` | `chillPy.plotting` | stub |
| `make_daily_chill_plot2` | `make_daily_chill_plot2` | `chillPy.plotting` | stub |
| `make_hourly_temps` | `make_hourly_temps` | `chillPy.temperature` | implemented |
| `make_multi_pheno_trend_plot` | `make_multi_pheno_trend_plot` | `chillPy.plotting` | stub |
| `make_pheno_trend_plot` | `make_pheno_trend_plot` | `chillPy.plotting` | stub |
| `ordered_climate_list` | `ordered_climate_list` | `chillPy.scenarios` | partial |
| `patch_daily_temperatures` | `patch_daily_temperatures` | `chillPy.temperature` | implemented |
| `patch_daily_temps` | `patch_daily_temps` | `chillPy.temperature` | implemented |
| `phenologyFit` | `phenology_fit` | `chillPy.phenology` | implemented |
| `phenologyFitter` | `phenology_fitter` | `chillPy.phenology` | partial |
| `plot_PLS` | `plot_pls` | `chillPy.plotting` | stub |
| `plot_climate_scenarios` | `plot_climate_scenarios` | `chillPy.plotting` | stub |
| `plot_phenology_trends` | `plot_phenology_trends` | `chillPy.plotting` | stub |
| `plot_scenarios` | `plot_scenarios` | `chillPy.plotting` | stub |
| `read_tab` | `read_tab` | `chillPy.utils` | stub |
| `runn_mean` | `runn_mean` | `chillPy.utils` | implemented |
| `runn_mean_pred` | `runn_mean_pred` | `chillPy.utils` | implemented |
| `save_temperature_scenarios` | `save_temperature_scenarios` | `chillPy.scenarios` | stub |
| `select_by_file_extension` | `select_by_file_extension` | `chillPy.utils` | partial |
| `stack_hourly_temps` | `stack_hourly_temps` | `chillPy.temperature` | implemented |
| `stage_transitions` | `stage_transitions` | `chillPy.phenology` | implemented |
| `step_model` | `step_model` | `chillPy.temperature_models` | implemented |
| `tempResponse` | `temp_response` | `chillPy.temperature` | implemented |
| `tempResponse_daily_list` | `temp_response_daily_list` | `chillPy.temperature` | partial |
| `tempResponse_hourtable` | `temp_response_hourtable` | `chillPy.temperature` | implemented |
| `temperature_generation` | `temperature_generation` | `chillPy.scenarios` | stub |
| `temperature_scenario_baseline_adjustment` | `temperature_scenario_baseline_adjustment` | `chillPy.scenarios` | stub |
| `temperature_scenario_from_records` | `temperature_scenario_from_records` | `chillPy.scenarios` | stub |
| `test_if_equal` | `test_if_equal` | `chillPy.utils` | partial |
| `weather2chillR` | `weather_to_chillr` | `chillPy.weather` | implemented |

## Implementation Notes

- `step_model`: R can return a list-like result when a temperature does not
  match any interval, for example `NA`. Python keeps the public result numeric;
  missing inputs become `nan`, and finite values outside all intervals raise
  `ValueError`.
- `Dynamic_Model` / `DynModel_driver`: missing temperatures raise `ValueError`.
  The R code does not define a robust missing-value path for these recurrences.
- `tempResponse_daily_list`: the idealized hourly-temperature path through
  `stack_hourly_temps` is implemented. The empirical path remains unavailable
  until `Empirical_hourly_temperatures` is ported.
- `tempResponse_hourtable`: when a season lacks an exact `start_jday` row,
  Python offsets that season at its first available row instead of propagating
  `NA` values from R's empty-index behavior.
- `make_all_day_table`: date completion, duplicate aggregation, hourly
  completion, and hourly-to-daily aggregation are implemented with pandas.
  Python raises `ValueError` for invalid timestep combinations where R returns
  warnings, and dates are timezone-naive despite accepting `tz` for API parity.
- `patch_daily_temperatures`: the deprecated daily patcher is implemented.
- `patch_daily_temps`: interval-specific bias correction is implemented for
  day, week, and month intervals. Python uses explicit `ValueError`s for
  invalid interval strings and returns sorted output; R can emit warnings or
  inherit merge ordering in some edge cases.
- `interpolate_gaps_hourly`: the Linvill-guided hourly interpolation workflow
  is implemented, including solved daily extremes, optional daily proxy
  patching via `patch_daily_temperatures`, interpolation of remaining daily
  extremes, and final deviation interpolation. Python validates invalid
  date/hour fields explicitly and uses `numpy.linalg.lstsq`; the R source uses
  `qr.solve` and contains an ambiguous negated count check around
  `minimum_values_for_solving`, so Python follows the documented threshold
  behavior.
- Date helpers: `make_JDay`, `add_date`, `Date2YEARMODA`, `YEARMODA2Date`,
  `JDay_earlier`, `JDay_later`, `JDay_count`, `get_last_date`, and
  `leap_year` are implemented. Python accepts vector-like inputs for scalar R
  helpers where useful and validates invalid finite dates explicitly; `make_JDay`
  also accepts a `Date` column as a Python convenience, while preserving the R
  behavior of row order and duplicate records for `Year`/`Month`/`Day` tables.
- `check_temperature_record`: daily and hourly format checks are implemented,
  including `YEARMODA`/`YEARMODAHO` expansion, missing and repeated record
  counts, and missing temperature counts. Python returns the R-style fields and
  also includes `valid` and `warnings` convenience fields.
- `fix_weather`: daily record completion, interpolation of selected columns,
  `no_<column>` missing-value flags, and seasonal QC summaries are implemented.
  Python raises explicit `ValueError`s for invalid schemas/dates; duplicate
  daily records are aggregated by the underlying `make_all_day_table` helper.
- `weather2chillR`: deterministic local conversion paths are implemented for
  GSOD, CIMIS, and UCIPM-style downloaded tables. Network download/list/delete
  handler actions remain placeholders. GSOD conversion follows the handler's
  full-year completion behavior; CIMIS/UCIPM conversions normalize columns,
  parse dates when needed, sort records, and aggregate duplicate dates.
- `chile_agromet2chillR`: Chile Agromet HTML/table conversion is implemented
  without adding XML dependencies. Python accepts either a path to an HTML table
  or an already-loaded DataFrame, validates malformed dates explicitly, and
  normalizes decimal-comma numeric fields.
- `genSeason` / `genSeasonList`: cross-year phenology season selection is
  implemented. `genSeason` returns zero-based `iloc` indices instead of R's
  one-based row numbers; `genSeasonList` preserves the R output columns
  `Temp`, `JDay`, and `Year`.
- `stage_transitions`: transition table construction and model accumulation
  through `tempResponse(..., whole_record=True)` are implemented. Python sorts
  hourly temperatures by `Year`, `JDay`, and `Hour` before slicing and raises
  explicit `ValueError`s for unknown stage names or invalid schemas; missing or
  duplicate observations leave transition metrics as `NaN`, matching the R
  control flow.
- `VIP`: implemented for single-response orthogonal-score PLS models. Python
  returns a DataFrame with component rows and predictor columns instead of an R
  matrix.
- `PLS_pheno` / `PLS_chill_force`: the non-plotting PLS data preparation,
  coefficient extraction, VIP scoring, and R-style summary tables are
  implemented with a small internal single-response PLS routine, avoiding a
  scikit-learn dependency. The returned `PLS_output` is a lightweight Python
  model object rather than an R `mvr` object. Cross-validation options other
  than `"none"` are not implemented.
- `color_bar_maker`: the deterministic threshold/sign color assignment used by
  R `plot_PLS` is implemented without graphics dependencies.
- `plot_PLS`: rendering remains a stub. Python provides
  `prepare_pls_plot_data` as a non-rendering helper that validates PLS outputs,
  classifies important rows by VIP threshold and coefficient sign, extracts
  contiguous important windows, and prepares bloom/chill/heat overlay tables for
  a future plotting layer.
- `bloom_prediction`, `bloom_prediction2`, and `bloom_prediction3`: sequential
  chill-then-heat prediction is implemented for scalar, paired vector,
  permutation, and hourly-temperature workflows. Python returns pandas
  DataFrames, accepts R-style keyword aliases such as `Chill_req` and
  `Start_JDay`, sorts inputs chronologically, and uses explicit `ValueError`s
  for invalid schemas or parameters. The R distinction is preserved where
  `bloom_prediction` reports the row after threshold crossings while
  `bloom_prediction2`/`3` report the current-row Julian date used by the
  vectorized R implementation.
- `UnifiedModel_Wrapper`, `UniChill_Wrapper`, `UniForce_Wrapper`, and
  `StepChill_Wrapper`: deterministic Chuine-style wrapper evaluations are
  implemented for season tables with `Temp` and `JDay` columns. Python returns
  `numpy.nan` for unmet chill/force requirements, raises explicit `ValueError`s
  for invalid parameter vectors or incomplete inputs, and preserves the R
  source's relative final `JDay` lookup after forcing starts. `StepChill_Wrapper`
  validates the five parameters used by the R source even though the Rd file
  describes seven parameters.
- `phenologyFit` / `phenologyFitter`: the R-style fit object, season
  validation, cross-year JDay unwrapping, wrapper-based predictions, NA
  prediction penalty, residual table, RMSE, and bounded parameter fitting are
  implemented. Python uses a deterministic bounded coordinate search instead of
  R's stochastic `GenSA`, avoiding a new optimizer dependency. The R default
  PhenoFlex wrapper remains unavailable until the PhenoFlex wrappers are ported,
  so `phenologyFitter` defaults to `UniForce_Wrapper` and is marked partial.
