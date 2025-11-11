import sys
import time
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QComboBox,
    QLabel,
    QDateEdit,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QWidget,
    QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QDate
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from windows_app.app_utils import load_data_from_sqlite, create_prediction, load_max_date, calculate_dates
from pandas.core.indexes.datetimes import DatetimeIndex

MAX_DATE = load_max_date("data_pipeline/stock_data.db")

start_qdate = QDate(MAX_DATE.year, MAX_DATE.month, MAX_DATE.day)

end_date = MAX_DATE + pd.Timedelta(days=30)

end_qdate = QDate(end_date.year, end_date.month, end_date.day)

class Worker(QThread):

    finished = pyqtSignal(np.ndarray, float, str, DatetimeIndex)
    error_occurred = pyqtSignal(str)

    def __init__(self, selected_stock, start_date, end_date):
        super().__init__()
        self.selected_stock = selected_stock
        self.start_date = start_date
        self.end_date = end_date
        self.DB_PATH = "data_pipeline/stock_data.db"
        self.MODEL_PATH = "your_model.joblib"
        self.SCALER_PATH = "your_scaler.joblib"

    def run(self):

        try:
           
            data = load_data_from_sqlite(self.DB_PATH, self.start_date, self.end_date)

            predictions = create_prediction(data)
            
            last_sentiment = data["Sentiment_labels"].values[-1]

            dates = calculate_dates(self.start_date, self.end_date)

            self.finished.emit(predictions[:len(dates)], data["Close"][-1:], last_sentiment, dates)

        except Exception as e:
            print(e)
            self.error_occurred.emit(f"An error occurred in the worker: {e}")


class MainWindow(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.setWindowTitle("Stock Prediction Application")  
        self.setWindowIcon(QIcon('generated_00.png')) 
        self.setGeometry(100, 100, 800, 600)  

        self.main_widget = QWidget(self)  
        self.setCentralWidget(self.main_widget)  
        self.main_layout = QVBoxLayout(self.main_widget)  
        self.main_layout.setContentsMargins(20, 20, 20, 20)  
        self.main_layout.setSpacing(20)  

        self.instruction_label = QLabel("Please select a stock and a date range for the prediction.", self)  
        self.instruction_label.setWordWrap(True)
        self.main_layout.addWidget(self.instruction_label)

        self.controls_frame = QFrame(self)  
        self.main_layout.addWidget(self.controls_frame)  
        self.controls_layout = QHBoxLayout(self.controls_frame)  
        self.controls_layout.setContentsMargins(0, 0, 0, 0) 
        self.controls_layout.setSpacing(10)  

        # Stock selection
        self.stock_label = QLabel("Select Stock:", self.controls_frame)  
        self.controls_layout.addWidget(self.stock_label)  
        stocks = ["OTP"] 
        self.stock_combobox = QComboBox(self.controls_frame)  
        self.stock_combobox.addItems(stocks)  
        self.controls_layout.addWidget(self.stock_combobox)  

        # Start Date
        self.start_date_label = QLabel("Start Date:", self.controls_frame)  
        self.controls_layout.addWidget(self.start_date_label)  
        self.start_date_entry = QDateEdit(self.controls_frame)  
        self.start_date_entry.setCalendarPopup(True)  
        self.start_date_entry.setDate(start_qdate)  
        self.controls_layout.addWidget(self.start_date_entry)  

        # End Date
        self.end_date_label = QLabel("End Date:", self.controls_frame)  
        self.controls_layout.addWidget(self.end_date_label)  
        self.end_date_entry = QDateEdit(self.controls_frame)  
        self.end_date_entry.setCalendarPopup(True)  
        self.end_date_entry.setDate(end_qdate)  
        self.controls_layout.addWidget(self.end_date_entry)  

        # Predict Button
        self.predict_button = QPushButton("Create Prediction", self.controls_frame)  
        self.predict_button.clicked.connect(self.create_prediction)  
        self.controls_layout.addWidget(self.predict_button)  

        self.loading_frame = QFrame(self)  
        self.loading_frame.setVisible(False)  
        
        loading_layout = QVBoxLayout(self.loading_frame)
        loading_layout.setAlignment(Qt.AlignCenter)
        self.loading_label = QLabel("Loading, please wait...", self.loading_frame)
        self.loading_label.setAlignment(Qt.AlignCenter)
        loading_layout.addWidget(self.loading_label)  
        
        self.main_layout.addWidget(self.loading_frame, alignment=Qt.AlignCenter)

        self.plot_frame = QFrame(self)  
        self.plot_layout = QVBoxLayout(self.plot_frame)  
        self.plot_frame.setLayout(self.plot_layout)  
        self.main_layout.addWidget(self.plot_frame)  

        self.last_price_label = QLabel("", self)  
        self.main_layout.addWidget(self.last_price_label)  

        self.sentiment_label = QLabel("", self)  
        self.main_layout.addWidget(self.sentiment_label)  

        self.main_layout.addStretch(1)

        self.worker = None

        try:
            with open("windows_app/style.qss", "r") as file:  
                self.setStyleSheet(file.read())  
        except FileNotFoundError:
            print("Warning: 'style.qss' not found. Using default styles.")
        except Exception as e:
            print(f"Warning: Could not load styles. {e}")


    def create_prediction(self):  

        selected_stock = self.stock_combobox.currentText()  
        start_date = self.start_date_entry.date().toPyDate()  
        end_date = self.end_date_entry.date().toPyDate()  

        if start_date >= end_date:  
            QMessageBox.warning(self, "Warning", "Start date must be before end date.")  
            return  

        self.loading_frame.setVisible(True) 
        self.predict_button.setEnabled(False)
        self.clear_plot_and_labels()

        self.worker = Worker(selected_stock, start_date, end_date)
        
        self.worker.finished.connect(self.display_results)
        self.worker.error_occurred.connect(self.handle_error)
        
        self.worker.start()

    def clear_plot_and_labels(self):

        for i in reversed(range(self.plot_layout.count())):  
            widget = self.plot_layout.itemAt(i).widget()  
            if widget:  
                widget.deleteLater()  

        self.last_price_label.setText("")  
        self.sentiment_label.setText("")

    def display_results(self, predicted_prices, last_known_price, sentiment, dates):

        self.loading_frame.setVisible(False) 
        self.predict_button.setEnabled(True)  
        selected_stock = self.stock_combobox.currentText()  

        try:
            plt.style.use('seaborn-v0_8-darkgrid') 
            fig, ax = plt.subplots(figsize=(10, 5))  
            
            # Plot the simple list of prices
            ax.plot(dates, predicted_prices, label='Előrejelzett értékek')

            ax.set_xlabel('Days')  
            ax.set_ylabel('Price')  
            ax.set_title(f'Előrejelzett értékek: {selected_stock}', fontsize=16)  
            ax.legend()  
            fig.autofmt_xdate(rotation=45, ha='right')
    
            canvas = FigureCanvas(fig)  
            self.plot_layout.addWidget(canvas)  
            canvas.draw()  


            last_predicted_price = predicted_prices[-1]  
            price_diff = last_predicted_price - last_known_price  
            price_diff_text = "increased" if price_diff > 0 else "decreased"  
            
            self.last_price_label.setText(
                f"Last Predicted Price: {last_predicted_price:.2f} "
                f"({price_diff_text} by {abs(price_diff):.2f} from last known price)"
            )  
            self.sentiment_label.setText(
                f"Sentiment of the latest news regarding {selected_stock} is {sentiment}."
            )
        
        except Exception as e:
            self.handle_error(f"Error displaying plot: {e}")


    def handle_error(self, error_message):
        """
        Called when the worker thread emits the 'error_occurred' signal.
        """
        self.loading_frame.setVisible(False)  # Hide loading
        self.predict_button.setEnabled(True)  # Re-enable button
        QMessageBox.critical(self, "Error", error_message)  

# --- Main execution ---
def main():  
    app = QApplication(sys.argv)  
    window = MainWindow()  
    window.show()  
    sys.exit(app.exec_())  

if __name__ == "__main__":  
    main()