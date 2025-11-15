import pytest
import sqlite3
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) 

from windows_app import app_utils 


def test_calculate_dates_normal_range():

    start = '2023-01-01' 
    end = '2023-01-07'   
    
    expected_dates = pd.to_datetime([
        '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'
    ])
    
    result = app_utils.calculate_dates(start, end)
    
    pd.testing.assert_index_equal(result, expected_dates)

def test_calculate_dates_no_business_days():
    start = '2023-01-06' 
    end = '2023-01-08'   
    
    expected_dates = pd.to_datetime([]) 
    
    result = app_utils.calculate_dates(start, end)
    
    pd.testing.assert_index_equal(result, expected_dates)

def test_calculate_dates_same_day():
    start = '2023-01-02' 
    end = '2023-01-02'   
    
    expected_dates = pd.to_datetime([])
    
    result = app_utils.calculate_dates(start, end)
    
    pd.testing.assert_index_equal(result, expected_dates)




@pytest.fixture
def in_memory_db():

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE OTP_BD (
        Date TEXT PRIMARY KEY,
        Close REAL,
        sentiment_score INTEGER
    )
    """)
    
    test_data = [
        ('2023-01-01', 100.0, 0),
        ('2023-01-02', 101.0, 1), 
        ('2023-01-03', 102.0, 2), 
        ('2023-01-04', 103.0, 0), 
    ]
    cursor.executemany("INSERT INTO OTP_BD VALUES (?, ?, ?)", test_data)
    conn.commit()
    
    yield conn
    
    conn.close()

@patch('windows_app.app_utils.sqlite3.connect') 
def test_load_max_date(mock_sqlite_connect, in_memory_db):

    mock_sqlite_connect.return_value = in_memory_db
    
    max_date = app_utils.load_max_date(db_path="fake/path/doesnt/matter")
    
    assert max_date == pd.to_datetime('2023-01-04')
    mock_sqlite_connect.assert_called_with("fake/path/doesnt/matter")


@patch('windows_app.app_utils.sqlite3.connect') 
def test_load_data_from_sqlite(mock_sqlite_connect, in_memory_db):

    mock_sqlite_connect.return_value = in_memory_db
    start_date = date(2023, 1, 3) 
    end_date = date(2023, 1, 5)  
    
    df = app_utils.load_data_from_sqlite(
        db_path="fake/path", 
        start_date=start_date, 
        end_date=end_date
    )
    
    assert len(df) == 3

    expected_sentiments = ['positive', 'negative', 'neutral']
    pd.testing.assert_series_equal(
        df['Sentiment_labels'], 
        pd.Series(expected_sentiments, name='Sentiment_labels')
    )
    
    assert df['Close'].iloc[0] == 100.0
    assert df['Close'].iloc[2] == 102.0


@patch('windows_app.app_utils.torch.load')        
@patch('windows_app.app_utils.NHiTS')            
@patch('windows_app.app_utils.TimeSeriesDataSet') 
@patch('windows_app.app_utils.SCALER')            
def test_create_prediction(mock_scaler, mock_timeseries_dataset, mock_nhits, mock_torch_load):

    fake_unscaled_preds = np.array([[150.0], [151.0], [152.0]])
    mock_scaler.inverse_transform.return_value = fake_unscaled_preds
    
    mock_model = MagicMock(name="MockNHiTSModel")
    
    fake_scaled_preds = "fake_torch_tensor_output" 
    mock_model.predict.return_value = fake_scaled_preds
    
    mock_nhits.from_dataset.return_value = mock_model
    
    mock_dataloader = MagicMock(name="MockDataloader")
    
    mock_ts_dataset_instance = MagicMock(name="MockTSDataset")
    mock_ts_dataset_instance.to_dataloader.return_value = mock_dataloader
    mock_timeseries_dataset.from_dataset.return_value = mock_ts_dataset_instance

    mock_torch_load.return_value = None 

    stock_data = pd.DataFrame({
        'Close': np.random.rand(70), 
        'Scaled_price': np.random.rand(70),
        'Sentiment_labels': ['positive'] * 70,
    })

    predictions = app_utils.create_prediction(stock_data)

    
    mock_nhits.from_dataset.assert_called_once()
    
    
    mock_model.predict.assert_called_with(mock_dataloader)
    
    mock_scaler.inverse_transform.assert_called_with(fake_scaled_preds)
    
    np.testing.assert_array_equal(predictions, fake_unscaled_preds.flatten())