import pathlib
import os
import yfinance as yf

UTILS_DIR = pathlib.Path(__file__).parent.absolute()

DATA_PATH = os.path.join(UTILS_DIR, os.pardir, os.pardir, 'data')
QUARTERLY_DB_NAME = 'quarterly_financial_data.db'
STOCKPUP_TABLE_NAME = 'stockpup_financial_data'
YF_QUARTERLY_TABLE_NAME = 'yahoo_financial_data'
QUARTERLY_DB_FILE_PATH = os.path.join(DATA_PATH, QUARTERLY_DB_NAME)
STOCK_GENERAL_INFO_CSV = os.path.join(DATA_PATH, 'stock_general_info.csv')

AVG_REC_PREFIX = 'AvgRecommendation_'

MARKET_INDICES = ['DJI', 'VTSAX', '^IXIC', '^GSPC']


class StockPupColumns:
    """
    Our dataset comes from over 20 years of 10-Q and 10-K filings made by public companies
     with the U.S. Securities and Exchange Commission. We extract data from both text and
     XBRL filings, fix reporting mistakes, and normalize the data into quarterly time series
     of final restated values.
    """
    # Date Quarter Ends
    QUARTER_END = "Quarter end"
    # The total number of common shares outstanding at the end of a given quarter, including all
    # classes of common stock.
    SHARES = "Shares"
    # The number of shares the company had at the end of a given quarter, adjusted for splits to
    # be comparable to today's shares.
    SHARES_SPLIT_ADJUSTED = "Shares split adjusted"
    # If an investor started with 1 share of stock at the end of a given quarter, the split factor
    # for that quarter indicates how many shares the investor would own today as a result of
    # subsequent stock splits.
    SPLIT_FACTOR = "Split factor"
    # Total assets at the end of a quarter.
    ASSETS = "Assets"
    # Current assets at the end of a quarter.
    CURRENT_ASSETS = "Current Assets"
    # Total liabilities at the end of a quarter.
    LIABILITIES = "Liabilities"
    # Current liabilities at the end of a quarter.
    CURRENT_LIABILITIES = "Current Liabilities"
    # Total shareholders' equity at the end of a quarter, including both common and preferred
    # stockholders.
    SHAREHOLDER_EQUITY = "Shareholders equity"
    # Non-controlling or minority interest, if any, excluded from Shareholders equity.
    NON_CONTROLLING_INTEREST = "Non-controlling interest"
    # Preferred equity, if any, included in Shareholders equity.
    PREFERRED_EQUITY = "Preferred equity"
    # Total Goodwill and all other Intangible assets, if any.
    GOODWILL_AND_INTANGIBLES = "Goodwill & intangibles"
    # All long-term debt including capital lease obligations.
    LONG_TERM_DEBT = "Long-term debt"
    # Total revenue for a given quarter.
    REVENUE = "Revenue"
    # Earnings or Net Income for a given quarter.
    EARNINGS = "Earnings"
    # Earnings available for common stockholders - Net income minus earnings that must be
    # distributed to preferred shareholders. May be omitted when not reported by the company.
    EARNINGS_AVAILABLE_FOR_COMMON_STOCKHOLDERS = "Earnings available for common stockholders"
    # Basic earnings per share for a given quarter.
    EPS_BASIC = "EPS basic"
    # Diluted earnings per share.
    EPS_DILUTED = "EPS diluted"
    # Common stock dividends paid during a quarter per share, including all regular and special
    # dividends and distributions to common shareholders.
    DIVIDEND_PER_SHARE = "Dividend per share"
    # Cash produced by operating activities during a given quarter, including Continuing and
    # Discontinued operations.
    CASH_FROM_OPERATING_ACTIVITES = "Cash from operating activities"
    # Cash produced by investing activities during a given quarter, including Continuing and
    # Discontinued operations.
    CASH_FROM_INVESTING_ACTIVITIES = "Cash from investing activities"
    # Cash produced by financing activities during a given quarter, including Continuing and
    # Discontinued operations.
    CASH_FROM_FINANCING_ACTIVITES = "Cash from financing activities"
    # Change in cash and cash equivalents during a given quarter, including Effect of Exchange
    # Rate Movements and Other Cash Change Adjustments, if any.
    CASH_CHANGE_DURING_PERIOD = "Cash change during period"
    # Cash and cash equivalents at the end of a quarter, including Continuing and
    # Discontinued operations.
    CASH_AT_END_OF_PERIOD = "Cash at end of period"
    # Capital Expenditures are the cash outflows for long-term productive assets, net of cash
    # from disposals of capital assets.
    CAPITAL_EXPENDITURES = "Capital expenditures"
    # The medium price per share of the company common stock during a given quarter. The prices
    # are as reported, and are not adjusted for subsequent dividends.
    PRICE = "Price"  # Average price during quarter
    # The highest price per share of the company common stock during a given quarter.
    PRICE_HIGH = "Price high"
    # The lowest price of the company common stock during a quarter.
    PRICE_LOW = "Price low"
    # Return on equity is the ratio of Earnings (available to common stockholders)
    # TTM (over the Trailing Twelve Months) to TTM average common shareholders' equity.
    ROE = "ROE"
    # Return on assets is the ratio of total Earnings TTM to TTM average Assets.
    ROA = "ROA"
    # Common stockholders' equity per share, also known as BVPS.
    BOOK_VALUE_OF_EQUITY_PER_SHARE = "Book value of equity per share"
    # The ratio of Price to Book value of equity per share as of the previous quarter.
    P_B_RATIO = "P/B ratio"
    # The ratio of Price to EPS diluted TTM as of the previous quarter.
    P_E_RATIO = "P/E ratio"
    # The aggregate amount of dividends paid per split-adjusted share of common stock from the
    # first available reporting quarter until a given quarter.
    CUM_DIVIDENDS_PER_SHARE = "Cumulative dividends per share"
    # The ratio of Dividends TTM to Earnings (available to common stockholders) TTM.
    DIVIDEND_PAYOUT_RATIO = "Dividend payout ratio"
    # The ratio of Long-term debt to common shareholders' equity (Shareholders equity minus
    # Preferred equity).
    LONG_TERM_DEBT_TO_EQUITY_RATIO = "Long-term debt to equity ratio"
    # The ratio of common shareholders' equity (Shareholders equity minus Preferred equity) to
    # Assets.
    EQUITY_TO_ASSETS_RATIO = "Equity to assets ratio"
    # The ratio of Earnings (available for common stockholders) TTM to Revenue TTM.
    NET_MARGIN = "Net margin"
    # The ratio of Revenue TTM to TTM average Assets.
    ASSET_TURNOVER = "Asset turnover"
    # Cash from operating activities minus the Capital Expenditures for a quarter.
    FREE_CASH_FLOW_PER_SHARE = "Free cash flow per share"
    # The ratio of Current assets to Current liabilities.
    CURRENT_RATIO = "Current ratio"


class QuarterlyColumns:
    TICKER_SYMBOL = "TickerSymbol"
    PRICE_AVG = "PriceAvg"
    PRICE_HI = "PriceHigh"
    PRICE_LO = "PriceLow"
    PRICE_AT_END_OF_QUARTER = "PriceEoQ"
    DATE = "Date"
    SHARES = "Shares"
    VOLUME = "Volume"
    ASSETS = "Assets"
    LIABILITIES = "Liabilities"
    LONG_TERM_DEBT = "LongTermDebt"
    REVENUE = "Revenue"
    EARNINGS = "Earnings"
    DIVIDENDS_PER_SHARE = "DividendsPerShare"
    ASSET_TURNOVER = "AssetTurnover"
    EPS = "EarningsPerShare"
    P_E_RATIO = "PriceToEarningsRatio"
    ROE = "ROE"
    ROA = "ROA"
    YEAR = "Year"
    QUARTER = "Quarter"
    SPLIT = "Split"
    AVG_RECOMMENDATIONS = "AvgRecommendations"
    EBIT = yf.FinancialColumns.Ebit.replace(' ', '')
    NET_INCOME = yf.FinancialColumns.NetIncome.replace(' ', '')
    GROSS_PROFIT = yf.FinancialColumns.GrossProfit.replace(' ', '')
    TOTAL_REVENUE = yf.FinancialColumns.TotalRevenue.replace(' ', '')
    RND = yf.FinancialColumns.RnD.replace(' ', '')
    OPERATING_EXPENSES = yf.FinancialColumns.TotalOperatingExpenses.replace(' ', '')
    INCOME_BEFORE_TAX = yf.FinancialColumns.IncomeBeforeTax.replace(' ', '')
    INCOME_TAX_EXPENSE = yf.FinancialColumns.IncomeTaxExpense.replace(' ', '')
    OPERATING_INCOME = yf.FinancialColumns.OperatingIncome.replace(' ', '')
    DIVIDENDS_PAID = yf.CashflowColumns.DividendsPaid.replace(' ', '')
    STOCK_REPURCHASED = yf.CashflowColumns.RepurchaseOfStock.replace(' ', '')
    DEPRECIATION = yf.CashflowColumns.Depreciation.replace(' ', '')
    STOCK_ISSUANCE = yf.CashflowColumns.IssuanceOfStock.replace(' ', '')
    NET_BORROWINGS = yf.CashflowColumns.NetBorrowings.replace(' ', '')
    INVESTMENTS = yf.CashflowColumns.Investments.replace(' ', '')
    CASH = yf.BalanceSheetColumns.Cash.replace(' ', '')
    STOCK = yf.BalanceSheetColumns.CommonStock.replace(' ', '')
    TOTAL_ASSETS = yf.BalanceSheetColumns.TotalAssets.replace(' ', '')
    TOTAL_LIABILITIES = yf.BalanceSheetColumns.TotalLiabilities.replace(' ', '')
    LONG_TERM_DEBT = yf.BalanceSheetColumns.LongTermDebt.replace(' ', '')
    SHORT_TERM_DEBT = yf.BalanceSheetColumns.ShortLongTermDebt.replace(' ', '')
    STOCKHOLDER_EQUITY = yf.BalanceSheetColumns.TotalStockholderEquity.replace(' ', '')
