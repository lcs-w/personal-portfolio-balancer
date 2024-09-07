import pandas as pd
import numpy as np

from datetime import datetime, date

from scipy.interpolate import CubicSpline

# load api key from a local .env file
import os
from dotenv import load_dotenv

load_dotenv()

from fredapi import Fred

# documentation: https://pypi.org/project/fredapi/

fred = Fred(api_key=os.getenv("fred_key"))


def get_interest_rate(
    sample_start: str = "1/1/2019",
    interpolated: bool = False,
    maturity: str = None,
) -> pd.DataFrame:

    t_dict = {
        "DGS1MO": (1 / 12),  # 1 month
        "DGS6MO": 0.5,  # 6 month
        "DGS1": 1,  # 1 year
        "DGS2": 2,
        "DGS3": 3,
        "DGS5": 5,  # 5 year
        "DGS7": 7,
        "DGS10": 10,  # 10 year
    }

    if maturity is not None:
        assert maturity in t_dict, "maturity must be one of {t}.".format(
            t=list(t_dict.keys())
        )
        t_dict = {maturity: t_dict[maturity]}

    t_mkt_yld = pd.DataFrame()
    for key in t_dict:
        _m = (
            fred.get_series(key, observation_start=sample_start).to_frame(
                t_dict[key] * 12
            )
            / 100
        )
        t_mkt_yld = t_mkt_yld.merge(_m, left_index=True, right_index=True, how="outer")

    t_mkt_yld = t_mkt_yld.dropna()
    t_mkt_yld = t_mkt_yld.reindex(sorted(t_mkt_yld.columns), axis=1)
    # t_mkt_yld.index += pd.offsets.MonthEnd(0)

    if interpolated:
        # cubic spline interpolation
        t_mkt_yld = pd.DataFrame(columns=range(1, 121))
        for _index, _row in t_mkt_yld.iterrows():
            _cs = CubicSpline(_row.index, _row)
            t_mkt_yld.loc[_index] = _cs(range(1, 121))

    return t_mkt_yld


def get_effective_rate(
    start_date: datetime = date(2019, 1, 1),
):
    sample_start = start_date.strftime("%Y-%m-%d")
    return (
        fred.get_series("DFF", observation_start=sample_start).to_frame(
            "effective_rate"
        )
        / 100
    )


if __name__ == "__name__":
    # t_mkt_yld = get_interest_rate()
    dff = get_effective_rate(date(2019, 10, 1))
    print(dff)
    pass
