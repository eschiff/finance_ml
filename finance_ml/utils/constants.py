import pathlib
import os

TICKER_SYMBOL, QUARTER, YEAR = 0, 1, 2
TARGET_COLUMN = 'PredictedAppreciation'

UTILS_DIR = pathlib.Path(__file__).parent.absolute()

DATA_PATH = os.path.join(UTILS_DIR, os.pardir, os.pardir, 'data')
QUARTERLY_DB_NAME = 'quarterly_financial_data.db'
STOCKPUP_TABLE_NAME = 'stockpup_data'
YF_QUARTERLY_TABLE_NAME = 'yahoo_financial_data'
QUARTERLY_DB_FILE_PATH = os.path.join(DATA_PATH, QUARTERLY_DB_NAME)
STOCK_GENERAL_INFO_CSV = os.path.join(DATA_PATH, 'stock_general_info.csv')

MISSING_SECTOR = 'MissingSector'
MISSING_INDUSTRY = 'MissingIndustry'

Q_DELTA_PREFIX = 'Q_Delta_'
YOY_DELTA_PREFIX = 'YOY_Delta_'
VS_MKT_IDX = '_vs_'
AVG_REC_SCORE_PREFIX = 'AvgRecScore_'

MARKET_INDICES = ['^DJI', 'VTSAX', '^IXIC', '^GSPC', '^RUT', '^NYA']

MONTH_TO_QUARTER = {
    1: 4,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 3,
    9: 3,
    10: 3,
    11: 4,
    12: 4
}


class StockPupColumns:
    """
    Our dataset comes from over 20 years of 10-Q and 10-K filings made by public companies
     with the U.S. Securities and Exchange Commission. We extract data from both text and
     XBRL filings, fix reporting mistakes, and normalize the data into quarterly time series
     of final restated values.
    """
    # Date Quarter Ends
    QUARTER_END = 'QuarterEnd'
    # The total number of common shares outstanding at the end of a given quarter, including all
    # classes of common stock.
    SHARES = 'Shares'
    # The number of shares the company had at the end of a given quarter, adjusted for splits to
    # be comparable to today's shares.
    SHARES_SPLIT_ADJUSTED = 'SharesSplitAdjusted'
    # If an investor started with 1 share of stock at the end of a given quarter, the split factor
    # for that quarter indicates how many shares the investor would own today as a result of
    # subsequent stock splits.
    SPLIT_FACTOR = 'SplitFactor'
    # Total assets at the end of a quarter.
    ASSETS = 'Assets'
    # Current assets at the end of a quarter.
    CURRENT_ASSETS = 'CurrentAssets'
    # Total liabilities at the end of a quarter.
    LIABILITIES = 'Liabilities'
    # Current liabilities at the end of a quarter.
    CURRENT_LIABILITIES = 'CurrentLiabilities'
    # Total shareholders' equity at the end of a quarter, including both common and preferred
    # stockholders.
    SHAREHOLDER_EQUITY = 'ShareholdersEquity'
    # Non-controlling or minority interest, if any, excluded from Shareholders equity.
    NON_CONTROLLING_INTEREST = 'NonControllingInterest'
    # Preferred equity, if any, included in Shareholders equity.
    PREFERRED_EQUITY = 'PreferredEquity'
    # Total Goodwill and all other Intangible assets, if any.
    GOODWILL_AND_INTANGIBLES = 'GoodwillIntangibles'
    # All long-term debt including capital lease obligations.
    LONG_TERM_DEBT = 'LongTermDebt'
    # Total revenue for a given quarter.
    REVENUE = 'Revenue'
    # Earnings or Net Income for a given quarter.
    EARNINGS = 'Earnings'
    # Earnings available for common stockholders - Net income minus earnings that must be
    # distributed to preferred shareholders. May be omitted when not reported by the company.
    EARNINGS_AVAILABLE_FOR_COMMON_STOCKHOLDERS = 'EarningsAvailableForCommonStockholders'
    # Basic earnings per share for a given quarter.
    EPS_BASIC = 'EPS_basic'
    # Diluted earnings per share.
    EPS_DILUTED = 'EPS_diluted'
    # Common stock dividends paid during a quarter per share, including all regular and special
    # dividends and distributions to common shareholders. Already adjusted for splits.
    DIVIDEND_PER_SHARE = 'DividendPerShare'
    # Cash produced by operating activities during a given quarter, including Continuing and
    # Discontinued operations.
    CASH_FROM_OPERATING_ACTIVITES = 'CashFromOperatingActivities'
    # Cash produced by investing activities during a given quarter, including Continuing and
    # Discontinued operations.
    CASH_FROM_INVESTING_ACTIVITIES = 'CashFromInvestingActivities'
    # Cash produced by financing activities during a given quarter, including Continuing and
    # Discontinued operations.
    CASH_FROM_FINANCING_ACTIVITES = 'CashFromFinancingActivities'
    # Change in cash and cash equivalents during a given quarter, including Effect of Exchange
    # Rate Movements and Other Cash Change Adjustments, if any.
    CASH_CHANGE_DURING_PERIOD = 'CashChangeDuringPeriod'
    # Cash and cash equivalents at the end of a quarter, including Continuing and
    # Discontinued operations.
    CASH_AT_END_OF_PERIOD = 'CashAtEndOfPeriod'
    # Capital Expenditures are the cash outflows for long-term productive assets, net of cash
    # from disposals of capital assets.
    CAPITAL_EXPENDITURES = 'CapitalExpenditures'
    # The medium price per share of the company common stock during a given quarter. The prices
    # are as reported, and are not adjusted for subsequent dividends.
    PRICE = 'Price'  # Average price during quarter
    # The highest price per share of the company common stock during a given quarter.
    PRICE_HIGH = 'PriceHigh'
    # The lowest price of the company common stock during a quarter.
    PRICE_LOW = 'PriceLow'
    # Return on equity is the ratio of Earnings (available to common stockholders)
    # TTM (over the Trailing Twelve Months) to TTM average common shareholders' equity.
    ROE = 'ROE'
    # Return on assets is the ratio of total Earnings TTM to TTM average Assets.
    ROA = 'ROA'
    # Common stockholders' equity per share, also known as BVPS.
    BOOK_VALUE_OF_EQUITY_PER_SHARE = 'BookValueOfEquityPerShare'
    # The ratio of Price to Book value of equity per share as of the previous quarter.
    P_B_RATIO = 'P_B_ratio'
    # The ratio of Price to EPS diluted TTM as of the previous quarter.
    P_E_RATIO = 'P_E_ratio'
    # The aggregate amount of dividends paid per split-adjusted share of common stock from the
    # first available reporting quarter until a given quarter.
    CUM_DIVIDENDS_PER_SHARE = 'CumulativeDividendsPerShare'
    # The ratio of Dividends TTM to Earnings (available to common stockholders) TTM.
    DIVIDEND_PAYOUT_RATIO = 'DividendPayoutRatio'
    # The ratio of Long-term debt to common shareholders' equity (Shareholders equity minus
    # Preferred equity).
    LONG_TERM_DEBT_TO_EQUITY_RATIO = 'LongTermDebtToEquityRatio'
    # The ratio of common shareholders' equity (Shareholders equity minus Preferred equity) to
    # Assets.
    EQUITY_TO_ASSETS_RATIO = 'EquityToAssetsRatio'
    # The ratio of Earnings (available for common stockholders) TTM to Revenue TTM.
    NET_MARGIN = 'NetMargin'
    # The ratio of Revenue TTM to TTM average Assets.
    ASSET_TURNOVER = 'AssetTurnover'
    # Cash from operating activities minus the Capital Expenditures for a quarter.
    FREE_CASH_FLOW_PER_SHARE = 'FreeCashFlowPerShare'
    # The ratio of Current assets to Current liabilities.
    CURRENT_RATIO = 'CurrentRatio'


class QuarterlyColumns:
    TICKER_SYMBOL = 'TickerSymbol'
    QUARTER = 'Quarter'
    YEAR = 'Year'
    PRICE_AVG = 'PriceAvg'
    PRICE_HI = 'PriceHigh'
    PRICE_LO = 'PriceLow'
    PRICE_AT_END_OF_QUARTER = 'PriceEoQ'
    AVG_RECOMMENDATIONS = 'AvgRecommendations'
    AVG_RECOMMENDATION_SCORE = 'AvgRecommendationScore'
    SPLIT = 'Split'
    EBIT = 'Ebit'
    PROFIT = 'GrossProfit'
    REVENUE = 'TotalRevenue'
    RND = 'ResearchDevelopment'
    OPERATING_EXPENSES = 'TotalOperatingExpenses'
    INCOME_PRETAX = 'IncomeBeforeTax'
    INCOME_TAX = 'IncomeTaxExpense'
    OPERATING_INCOME = 'OperatingIncome'
    NET_INCOME = 'NetIncome'
    DIVIDENDS = 'DividendsPaid'
    STOCK_REPURCHASED = 'RepurchaseOfStock'
    STOCK_ISSUED = 'IssuanceOfStock'
    DEPRECIATION = 'Depreciation'
    NET_BORROWINGS = 'NetBorrowings'
    INVESTMENTS = 'Investments'
    CASH = 'Cash'
    COMMON_STOCK = 'CommonStock'
    ASSETS = 'TotalAssets'
    LIABILITIES = 'TotalLiab'
    DEBT_LONG = 'LongTermDebt'
    DEBT_SHORT = 'ShortLongTermDebt'
    DATE = 'Date'
    VOLUME = 'Volume'
    EARNINGS = 'Earnings'
    STOCKHOLDER_EQUITY = 'TotalStockholderEquity'
    DIVIDEND_PER_SHARE = 'DividendPerShare'
    VOLATILITY = 'Volatility'
    SECTOR = 'Sector'
    INDUSTRY = 'Industry'
    MARKET_CAP = 'MarketCap'
    AGE_OF_DATA = 'AgeOfData'
    WORKING_CAPITAL_RATIO = 'AssetsToLiabilitiesRatio'
    AVG_PE_RATIO = 'AvgPriceToEarningsRatio'
    DEBT_EQUITY_RATIO = 'DebtToEquityRatio'
    ROE = 'ReturnOnEquity'
    PRICE_BOOK_RATIO = 'PriceToBookRatio'
    FCF = 'FreeCashFlow'
    PROFIT_MARGIN = 'ProfitMargin'
    RND_RATIO = "RnDtoRevenue"
    CASH_RATIO = "CashToRevenue"
    EXPENSES_RATIO = "ExpensesToRevenue"

    @staticmethod
    def columns():
        return [getattr(QuarterlyColumns, col) for col in dir(QuarterlyColumns) if
                col[0] != '_' and col != 'columns']


CATEGORICAL_COLUMNS = [
    QuarterlyColumns.SECTOR,
    QuarterlyColumns.INDUSTRY,
    QuarterlyColumns.QUARTER
]

NUMERIC_COLUMNS = [
    QuarterlyColumns.PRICE_AVG,
    QuarterlyColumns.PRICE_HI,
    QuarterlyColumns.PRICE_LO,
    QuarterlyColumns.PRICE_AT_END_OF_QUARTER,
    QuarterlyColumns.EBIT,
    QuarterlyColumns.PROFIT,
    QuarterlyColumns.REVENUE,
    QuarterlyColumns.RND,
    QuarterlyColumns.OPERATING_EXPENSES,
    QuarterlyColumns.INCOME_PRETAX,
    QuarterlyColumns.INCOME_TAX,
    QuarterlyColumns.OPERATING_INCOME,
    QuarterlyColumns.NET_INCOME,
    QuarterlyColumns.DIVIDENDS,
    QuarterlyColumns.STOCK_REPURCHASED,
    QuarterlyColumns.DEPRECIATION,
    QuarterlyColumns.STOCK_ISSUED,
    QuarterlyColumns.CASH,
    QuarterlyColumns.COMMON_STOCK,
    QuarterlyColumns.ASSETS,
    QuarterlyColumns.LIABILITIES,
    QuarterlyColumns.DEBT_LONG,
    QuarterlyColumns.DEBT_SHORT,
    QuarterlyColumns.STOCKHOLDER_EQUITY,
    QuarterlyColumns.VOLUME,
    QuarterlyColumns.EARNINGS,
    QuarterlyColumns.RND_RATIO,
    QuarterlyColumns.CASH_RATIO
]

COLUMNS_TO_COMPARE_TO_MARKET_INDICES = [
    f"{Q_DELTA_PREFIX}{QuarterlyColumns.PRICE_AVG}",
    f"{YOY_DELTA_PREFIX}{QuarterlyColumns.PRICE_AVG}",
    QuarterlyColumns.VOLATILITY,
]

FORMULAE = {
    QuarterlyColumns.VOLATILITY:
        lambda row: (row[QuarterlyColumns.PRICE_HI] - row[QuarterlyColumns.PRICE_LO]) / row[
            QuarterlyColumns.PRICE_AVG],

    QuarterlyColumns.WORKING_CAPITAL_RATIO:
        lambda row: (row[QuarterlyColumns.ASSETS] / row[QuarterlyColumns.LIABILITIES]),

    QuarterlyColumns.AVG_PE_RATIO: lambda row: (
            row[QuarterlyColumns.PRICE_AVG] / row[QuarterlyColumns.EARNINGS]),

    QuarterlyColumns.DEBT_EQUITY_RATIO:
        lambda row: (row[QuarterlyColumns.DEBT_LONG] + row[QuarterlyColumns.DEBT_SHORT]) / row[
            QuarterlyColumns.STOCKHOLDER_EQUITY],

    QuarterlyColumns.ROE:
        lambda row: (row[QuarterlyColumns.EARNINGS] - row[QuarterlyColumns.DIVIDENDS]) / row[
            QuarterlyColumns.STOCKHOLDER_EQUITY],

    QuarterlyColumns.PRICE_BOOK_RATIO:
        lambda row: (row[QuarterlyColumns.ASSETS] - row[QuarterlyColumns.LIABILITIES]) / row[
            QuarterlyColumns.MARKET_CAP],

    QuarterlyColumns.PROFIT_MARGIN:
        lambda row: (row[QuarterlyColumns.NET_INCOME] / row[QuarterlyColumns.REVENUE]),

    QuarterlyColumns.RND_RATIO:
        lambda row: (row[QuarterlyColumns.RND] / row[QuarterlyColumns.REVENUE]),

    QuarterlyColumns.CASH_RATIO:
        lambda row: (row[QuarterlyColumns.CASH] / row[QuarterlyColumns.REVENUE]),

    QuarterlyColumns.EXPENSES_RATIO:
        lambda row: (row[QuarterlyColumns.OPERATING_EXPENSES] / row[QuarterlyColumns.REVENUE])
}

INDEX_COLUMNS = [QuarterlyColumns.TICKER_SYMBOL,
                 QuarterlyColumns.QUARTER,
                 QuarterlyColumns.YEAR]
