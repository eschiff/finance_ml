from datetime import datetime, date
from finance_ml.utils.quarterly_index import QuarterlyIndex


def test_quarterly_index():
    q_idx = QuarterlyIndex('A', 1, 2020)

    assert q_idx.ticker == 'A'
    assert q_idx.quarter == 1
    assert q_idx.year == 2020
    assert q_idx.time_travel(1) == QuarterlyIndex('A', 2, 2020)
    assert q_idx.time_travel(4) == QuarterlyIndex('A', 1, 2021)
    assert q_idx.time_travel(-1) == QuarterlyIndex('A', 4, 2019)
    assert q_idx.time_travel(1) > q_idx
    assert q_idx < q_idx.time_travel(1)
    assert q_idx.to_tuple() == ('A', 1, 2020)
    assert q_idx.to_xQyyyy() == '1Q2020'
    assert q_idx.to_date() == datetime(year=2020, month=3, day=1).date()
