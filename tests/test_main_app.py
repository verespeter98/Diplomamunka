import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import numpy as np
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QDate, Qt

from pytestqt.qt_compat import qt_api

import main_app

@pytest.fixture
def app(qtbot):

    with patch('main_app.start_qdate', QDate(2023, 1, 1)), \
         patch('main_app.end_qdate', QDate(2023, 1, 31)):
        
        window = main_app.MainWindow()
        window.show()
        qtbot.addWidget(window)
        yield window
        
        window.close()

def test_main_window_initial_state(app):

    assert app.windowTitle() == "Pénzügyi előrejelzés"
    assert app.stock_combobox.currentText() == "OTP"
    assert app.start_date_entry.date() == QDate(2023, 1, 1)
    assert app.end_date_entry.date() == QDate(2023, 1, 31)
    assert not app.loading_frame.isVisible()
    assert app.predict_button.isEnabled()
    assert app.last_price_label.text() == ""
    assert app.sentiment_label.text() == ""
    assert app.plot_layout.count() == 0

def test_create_prediction_invalid_date(app, qtbot, monkeypatch):

    mock_warning = MagicMock()
    monkeypatch.setattr(main_app.QMessageBox, "warning", mock_warning)
    
    app.start_date_entry.setDate(QDate(2023, 2, 1))
    app.end_date_entry.setDate(QDate(2023, 1, 1))
    
    qtbot.mouseClick(app.predict_button, Qt.LeftButton)
    
    mock_warning.assert_called_once_with(app, "Warning", "A kezdő dátumnak a végdátumnál korábbi időpontnak kell lennie!")
    assert app.worker is None
    assert not app.loading_frame.isVisible()


@patch('main_app.load_data_from_sqlite', side_effect=Exception("Database connection failed"))
def test_create_prediction_worker_error(mock_load_data, app, qtbot, monkeypatch):

    mock_critical = MagicMock()
    monkeypatch.setattr(main_app.QMessageBox, "critical", mock_critical)
    
    qtbot.mouseClick(app.predict_button, Qt.LeftButton)
    
    assert app.loading_frame.isVisible()
    
    qtbot.waitUntil(lambda: not app.loading_frame.isVisible(), timeout=5000)

    assert not app.loading_frame.isVisible()
    assert app.predict_button.isEnabled()
    assert app.plot_layout.count() == 0
    
    mock_critical.assert_called_once()
    args, _ = mock_critical.call_args
    assert args[0] == app
    assert args[1] == "Error"
    assert "Database connection failed" in args[2]

@patch('main_app.load_data_from_sqlite')
@patch('main_app.create_prediction')
@patch('main_app.calculate_dates')
def test_worker_run_success(mock_calc_dates, mock_create_pred, mock_load_data, qtbot):

    mock_data = pd.DataFrame({'Sentiment_labels': ['Negative'], 'Close': [50.0]})
    mock_predictions = np.array([45.0, 40.0])
    mock_dates = pd.to_datetime(['2023-01-01', '2023-01-02'])

    mock_load_data.return_value = mock_data
    mock_create_pred.return_value = mock_predictions
    mock_calc_dates.return_value = mock_dates

    start_date_py = date(2023, 1, 1)
    end_date_py = date(2023, 1, 3)
    
    worker = main_app.Worker("OTP", start_date_py, end_date_py)

    with qtbot.waitSignal(worker.finished, timeout=1000) as blocker:
        worker.start()
    
    assert len(blocker.args) == 4
    emitted_preds, emitted_last_price, emitted_sentiment, emitted_dates = blocker.args
    
    np.testing.assert_array_equal(emitted_preds, np.array([45.0, 40.0]))
    assert emitted_last_price == 50.0
    assert emitted_sentiment == "Negative"
    pd.testing.assert_index_equal(emitted_dates, mock_dates)

@patch('main_app.load_data_from_sqlite', side_effect=Exception("Test error in worker"))
def test_worker_run_error(mock_load_data, qtbot):

    start_date_py = date(2023, 1, 1)
    end_date_py = date(2023, 1, 3)
    
    worker = main_app.Worker("OTP", start_date_py, end_date_py)

    with qtbot.waitSignal(worker.error_occurred, timeout=1000) as blocker:
        worker.start()

    assert len(blocker.args) == 1
    error_msg = blocker.args[0]
    assert "Test error in worker" in error_msg
    assert "An error occurred in the worker" in error_msg