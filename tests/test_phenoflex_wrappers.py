import numpy as np
import pandas as pd
import pytest
from chillPy.phenology import (
    phenoflex_gdh_wrapper,
    phenoflex_gauss_wrapper,
    phenoflex_fixed_dynamic_model_wrapper,
    phenoflex_fixed_dynamic_model_gauss_wrapper,
    _phenoflex_smooth,
    phenology_fitter
)

@pytest.fixture
def sample_season():
    # A simple season with 2000 hours of constant temperature
    # JDays from 305 to 365, then 1 to 100 (cross year)
    jdays_part1 = np.arange(305, 366)
    jdays_part2 = np.arange(1, 101)
    jdays_daily = np.concatenate([jdays_part1, jdays_part2])
    jdays = np.repeat(jdays_daily, 24)
    # R PhenoFlex expects temperature in Celsius if deg_celsius=True (default)
    # 5C for first 60 days (chill), then 25C (heat)
    # 25C is Tu for GDH, so maximum accumulation.
    temps = np.full(len(jdays), 5.0) 
    temps[60*24:] = 25.0
    return pd.DataFrame({"Temp": temps, "JDay": jdays})

def test_debug_phenoflex(sample_season):
    from chillPy.temperature_models import phenoflex
    par = [20, 1.0, 0.5, 25, 3372.8, 9900.3, 6319.5, 5.939917e13, 4, 36, 4, 1.6]
    res = phenoflex(
        temp=sample_season["Temp"].values,
        times=np.arange(1, len(sample_season) + 1),
        yc=par[0],
        zc=par[1],
        s1=par[2],
        tu=par[3],
        e0=par[4],
        e1=par[5],
        a0=par[6],
        a1=par[7],
        tf=par[8],
        tc=par[9],
        tb=par[10],
        slope=par[11],
        imodel=0,
        basic_output=False
    )
    # print for debug
    print(f"Bloom index: {res['bloomindex']}")
    print(f"Max y: {np.max(res['y'])}")
    print(f"Max z: {np.max(res['z'])}")
    assert res['bloomindex'] > 0

def test_phenoflex_smooth_logic():
    jdays = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    # Case 1: n = 1 (unique jday)
    assert _phenoflex_smooth(np.array([1, 2, 3]), 2) == 2.0
    
    # Case 2: n > 1
    # n=3, jday=2, jday_list=[3, 4, 5] (indices)
    # bloom_index=4 (pos 1), bloom_index=5 (pos 2), bloom_index=6 (pos 3)
    # Formula: JDay + pos/n - 1.0 / (n / ceil(n/2.0))
    # n=3, ceil(n/2)=2, factor = 1.0 / (3/2) = 2/3
    
    # bloomindex = 4 -> idx = 3. pos = 1. 2 + 1/3 - 2/3 = 2 - 1/3 = 1.666...
    assert _phenoflex_smooth(jdays, 4) == pytest.approx(1.666666666)
    # bloomindex = 5 -> idx = 4. pos = 2. 2 + 2/3 - 2/3 = 2.0
    assert _phenoflex_smooth(jdays, 5) == pytest.approx(2.0)
    # bloomindex = 6 -> idx = 5. pos = 3. 2 + 3/3 - 2/3 = 2 + 1/3 = 2.333...
    assert _phenoflex_smooth(jdays, 6) == pytest.approx(2.333333333)

def test_phenoflex_gdh_wrapper_valid(sample_season):
    # yc, zc, s1, Tu, E0, E1, A0, A1, Tf, Tc, Tb, slope
    # lower yc and zc to ensure bloom in synthetic data
    par = [20, 1.0, 0.5, 25, 3372.8, 9900.3, 6319.5, 5.939917e13, 4, 36, 4, 1.6]
    res = phenoflex_gdh_wrapper(sample_season, par)
    assert isinstance(res, float)
    assert not np.isnan(res)

def test_phenoflex_gauss_wrapper_valid(sample_season):
    # yc, zc, s1, Tu, E0, E1, A0, A1, Tf, Delta, slope
    par = [20, 1.0, 0.5, 25, 3372.8, 9900.3, 6319.5, 5.939917e13, 4, 4, 1.6]
    res = phenoflex_gauss_wrapper(sample_season, par)
    assert isinstance(res, float)
    assert not np.isnan(res)

def test_phenoflex_fixed_dynamic_wrapper_valid(sample_season):
    # yc, zc, s1, Tu, Tc, Tb
    par = [20, 1.0, 0.5, 25, 36, 4]
    res = phenoflex_fixed_dynamic_model_wrapper(sample_season, par)
    assert isinstance(res, float)
    assert not np.isnan(res)

def test_phenoflex_fixed_dynamic_gauss_wrapper_valid(sample_season):
    # yc, zc, s1, Tu, Delta
    par = [20, 1.0, 0.5, 25, 4]
    res = phenoflex_fixed_dynamic_model_gauss_wrapper(sample_season, par)
    assert isinstance(res, float)
    assert not np.isnan(res)

def test_phenoflex_wrappers_invalid_params(sample_season):
    # GDH Tb >= Tu should return NaN
    par_gdh = [40, 190, 0.5, 25, 3372.8, 9900.3, 6319.5, 5.939917e13, 4, 36, 26, 1.6]
    assert np.isnan(phenoflex_gdh_wrapper(sample_season, par_gdh))
    
    # GDH Tc <= Tu should return NaN
    par_gdh2 = [40, 190, 0.5, 25, 3372.8, 9900.3, 6319.5, 5.939917e13, 4, 24, 4, 1.6]
    assert np.isnan(phenoflex_gdh_wrapper(sample_season, par_gdh2))

def test_integration_with_phenology_fitter(sample_season):
    # Use 2 seasons for fitting
    season_list = [sample_season, sample_season]
    bloom_jdays = [15.0, 15.0]
    
    par_guess = [20, 1.0, 0.5, 25, 3372.8, 9900.3, 6319.5, 5.939917e13, 4, 36, 4, 1.6]
    lower = [18, 0.5, 0.1, 20, 3000, 9000, 6000, 5.e13, 0, 30, 0, 1.0]
    upper = [22, 2.0, 1.0, 30, 4000, 10000, 7000, 6.e13, 10, 40, 10, 2.0]
    
    # Run fitter for few iterations
    res = phenology_fitter(
        par_guess=par_guess,
        modelfn=phenoflex_gdh_wrapper,
        bloom_jdays=bloom_jdays,
        season_list=season_list,
        lower=lower,
        upper=upper,
        control={"maxit": 2}
    )
    
    assert res["par"] is not None
    assert len(res["pbloomJDays"]) == 2
