from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.transforms import QuarterFilter


def test_QuarterFilter(quarterly_df_with_predictions):
    QF = QuarterFilter(start_qtr=QuarterlyIndex('', 2, 2020),
                       end_qtr=QuarterlyIndex('', 2, 2021))

    output = QF.transform(quarterly_df_with_predictions)

    assert set(output.index) == {
        ('A', 2, 2020), ('A', 3, 2020), ('A', 4, 2020), ('A', 1, 2021),
        ('AAPL', 2, 2020), ('AAPL', 3, 2020), ('AAPL', 4, 2020), ('AAPL', 1, 2021)}
