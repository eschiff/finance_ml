import yfinance_ez as yf

from finance_ml.utils.constants import QuarterlyColumns

INFO_KEYS = [
    yf.TickerInfoKeys.sector,
    yf.TickerInfoKeys.marketCap,
    yf.TickerInfoKeys.shortName,
    yf.TickerInfoKeys.category,
    yf.TickerInfoKeys.industry,
]

FINANCIAL_KEYS = [
    yf.FinancialColumns.Ebit,
    yf.FinancialColumns.GrossProfit,
    yf.FinancialColumns.TotalRevenue,
    yf.FinancialColumns.RnD,
    yf.FinancialColumns.TotalOperatingExpenses,
    yf.FinancialColumns.IncomeBeforeTax,
    yf.FinancialColumns.IncomeTaxExpense,
    yf.FinancialColumns.OperatingIncome
]

CASHFLOW_KEYS = [
    yf.CashflowColumns.NetIncome,
    yf.CashflowColumns.DividendsPaid,
    yf.CashflowColumns.RepurchaseOfStock,
    yf.CashflowColumns.Depreciation,
    yf.CashflowColumns.IssuanceOfStock,
    yf.CashflowColumns.NetBorrowings,
    yf.CashflowColumns.Investments
]

BALANCE_SHEET_KEYS = [
    yf.BalanceSheetColumns.Cash,
    yf.BalanceSheetColumns.CommonStock,
    yf.BalanceSheetColumns.TotalAssets,
    yf.BalanceSheetColumns.TotalLiabilities,
    yf.BalanceSheetColumns.TotalStockholderEquity,
    yf.BalanceSheetColumns.LongTermDebt
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

YF_QUARTERLY_TABLE_SCHEMA = f'''
    {QuarterlyColumns.TICKER_SYMBOL} TEXT,
    {QuarterlyColumns.QUARTER} INT,
    {QuarterlyColumns.YEAR} INT,
    {QuarterlyColumns.PRICE_AVG} NUMERIC,
    {QuarterlyColumns.PRICE_HI} NUMERIC,
    {QuarterlyColumns.PRICE_LO} NUMERIC,
    {QuarterlyColumns.PRICE_AT_END_OF_QUARTER} NUMERIC,
    {QuarterlyColumns.AVG_RECOMMENDATIONS} TEXT,
    {QuarterlyColumns.SPLIT} NUMERIC,
    {QuarterlyColumns.EBIT} NUMERIC,
    {QuarterlyColumns.PROFIT} NUMERIC,
    {QuarterlyColumns.REVENUE} NUMERIC,
    {QuarterlyColumns.RND} NUMERIC,
    {QuarterlyColumns.OPERATING_EXPENSES} NUMERIC,
    {QuarterlyColumns.INCOME_PRETAX} NUMERIC,
    {QuarterlyColumns.INCOME_TAX} NUMERIC,
    {QuarterlyColumns.OPERATING_INCOME} NUMERIC,
    {QuarterlyColumns.NET_INCOME} NUMERIC,
    {QuarterlyColumns.DIVIDENDS} NUMERIC,
    {QuarterlyColumns.STOCK_REPURCHASED} NUMERIC,
    {QuarterlyColumns.DEPRECIATION} NUMERIC,
    {QuarterlyColumns.STOCK_ISSUED} NUMERIC,
    {QuarterlyColumns.NET_BORROWINGS} NUMERIC,
    {QuarterlyColumns.INVESTMENTS} NUMERIC,
    {QuarterlyColumns.CASH} NUMERIC,
    {QuarterlyColumns.COMMON_STOCK} NUMERIC,
    {QuarterlyColumns.ASSETS} NUMERIC,
    {QuarterlyColumns.LIABILITIES} NUMERIC,
    {QuarterlyColumns.DEBT_LONG} NUMERIC,
    {QuarterlyColumns.DATE} TEXT,
    {QuarterlyColumns.STOCKHOLDER_EQUITY} INT,
    {QuarterlyColumns.VOLUME} INT,
    {QuarterlyColumns.EARNINGS} INT,
    {QuarterlyColumns.DEBT_SHORT} NUMERIC,
    {QuarterlyColumns.MARKET_CAP} NUMERIC'''
