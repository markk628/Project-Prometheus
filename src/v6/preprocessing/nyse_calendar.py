"""
NYSE trading-day calendar helpers.

Named `nyse_calendar` rather than `calendar` to avoid confusion with the
Python stdlib `calendar` module. The only function here right now is
`_build_nyse_valid_days`, used by `_handle_gaps` to fill missing trading
days inside a per-ticker series.
"""

import pandas_market_calendars as mcal
import polars as pl


def _build_nyse_valid_days(start_date, end_date) -> pl.DataFrame:
    """
    Return a DataFrame of every valid NYSE trading day in the date range.

    Each trading day is represented by a single Date, used as the join key
    for gap-filling daily bars.
    """
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    trading_dates = [row.name.date() for _, row in schedule.iterrows()]

    return pl.DataFrame({"date": trading_dates}).with_columns(
        pl.col("date").cast(pl.Date)
    )
