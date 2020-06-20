import yfinance as yf

INFO_KEYS = [
    yf.TickerInfoKeys.sector,
    yf.TickerInfoKeys.marketCap,
    yf.TickerInfoKeys.shortName,
    yf.TickerInfoKeys.category,
]

FINANCIAL_KEYS = [
    yf.FinancialColumns.Ebit,
    yf.FinancialColumns.NetIncome,
    yf.FinancialColumns.GrossProfit,
    yf.FinancialColumns.TotalRevenue,
    yf.FinancialColumns.RnD,
    yf.FinancialColumns.TotalOperatingExpenses
]

CASHFLOW_KEYS = [
    yf.CashflowColumns.NetIncome,
    yf.CashflowColumns.DividendsPaid,
    yf.CashflowColumns.RepurchaseOfStock,
]

BALANCE_SHEET_KEYS = [
    yf.BalanceSheetColumns.Cash,
    yf.BalanceSheetColumns.CommonStock,
    yf.BalanceSheetColumns.TotalAssets,
    yf.BalanceSheetColumns.TotalLiabilities,
    yf.BalanceSheetColumns.LongTermDebt,
    yf.BalanceSheetColumns.ShortLongTermDebt
]

# Converting grades to positive, neutral, or negative
RECOMMENDATION_GRADE_MAPPING = {
    yf.RecommendationGrades.Buy: 1,
    yf.RecommendationGrades.Outperform: 1,
    yf.RecommendationGrades.Overweight: 1,
    yf.RecommendationGrades.SectorPerform: 0,
    yf.RecommendationGrades.Neutral: 0,
    yf.RecommendationGrades.SectorOutperform: 1,
    yf.RecommendationGrades.Hold: 0,
    yf.RecommendationGrades.MarketPerform: 0,
    yf.RecommendationGrades.StrongBuy: 1,
    yf.RecommendationGrades.LongTermBuy: 1,
    yf.RecommendationGrades.Sell: -1,
    yf.RecommendationGrades.MarketOutperform: 1,
    yf.RecommendationGrades.Positive: 1,
    yf.RecommendationGrades.EqualWeight: 0,
    yf.RecommendationGrades.Perform: 0,
    yf.RecommendationGrades.Negative: -1,
    yf.RecommendationGrades.SectorWeight: 0,
    yf.RecommendationGrades.Reduce: -1,
    yf.RecommendationGrades.Underweight: -1
}
