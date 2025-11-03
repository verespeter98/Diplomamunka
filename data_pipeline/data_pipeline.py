from get_news import TARGET_CLASSES
from get_news import load_news
from get_stock_prices import download_stock_data, normalize_stock_data
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def merge_results(stock_price_df, news_df):
    """
    Merges the aggregated sentiment DataFrame into an existing DataFrame on the 'date' column.
    Assumes existing_df has a 'date' column.
    Uses left join to preserve all rows in existing_df.
    """
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df["Date"] = stock_price_df["Date"].dt.tz_convert("Europe/Budapest")
    news_df["Date"] = news_df["Date"].dt.tz_localize("Europe/Budapest")
    merged = pd.merge(stock_price_df, news_df, on='Date', how='left')
    return merged

def load_to_sql(df, ticker, db_name="stock_data.db", table_name=None):
    """
    Load stock price data from a pandas DataFrame into a SQLite database.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing stock price data
    ticker (str): Stock ticker symbol (e.g., 'AAPL')
    db_name (str): Name of the SQLite database file (default: 'stock_data.db')
    table_name (str): Name of the table; if None, uses ticker as table name
    
    Returns:
    bool: True if data was loaded successfully, False otherwise
    """
    try:
        # Use ticker as table name if none provided
        if table_name is None:
            table_name = ticker.replace(".", "_")  # Replace invalid characters for table names
        
        # Connect to SQLite database (creates file if it doesn't exist)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """)
        
        # Convert DataFrame to records and insert into table
        # Ensure date is in string format for SQLite
        df['Date'] = df['Date'].astype(str)
        df.to_sql(table_name, conn, if_exists='append', index=False)
        
        # Commit changes and close connection
        conn.commit()
        print(f"Appended {len(df)} new rows to '{table_name}' in {db_name}")
        return True
    
    except Exception as e:
        print(f"Error loading data to SQLite: {str(e)}")
        return False
    
    finally:
        conn.close()



def main():

    ticker = "OTP.BD"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = "2000-01-01"

    stock_prices = download_stock_data(ticker, start_date, end_date)
    stock_prices = normalize_stock_data(stock_prices)

    stock_prices.to_csv(f"{ticker}_prices_saved.csv", index=False)

    stock_news = load_news(ticker, str(start_date), str(end_date))

    merged = merge_results(stock_prices, stock_news)
    
    # Load to SQLite
    load_to_sql(merged, ticker)

if __name__ == "__main__":
    main()