"""
Main GUI Window for Vibration Analyzer
Professional interface for loading data and performing analysis
"""

import sys
import os
import traceback
from typing import Optional

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QLabel, QFileDialog, QTextEdit, 
                           QTabWidget, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
                           QComboBox, QProgressBar, QMessageBox, QTableWidget, 
                           QTableWidgetItem, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Import our analysis modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.analysis.signal_processor import VibrationProcessor

class AnalysisWorker(QThread):
    """Worker thread for signal analysis to prevent GUI freezing"""
    
    analysis_completed = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, filepath: str, sampling_rate: float, rpm: Optional[float] = None):
        super().__init__()
        self.filepath = filepath
        self.sampling_rate = sampling_rate
        self.rpm = rpm
        
    def run(self):
        """Run analysis in background thread"""
        try:
            processor = VibrationProcessor(self.sampling_rate)
            
            self.progress_update.emit("Loading data...")
            time_data, signal_data = processor.load_csv_data(self.filepath)
            
            self.progress_update.emit("Performing time-domain analysis...")
            time_features = processor.time_domain_analysis(signal_data)
            
            self.progress_update.emit("Performing frequency-domain analysis...")
            freq_analysis = processor.frequency_domain_analysis(signal_data)
            
            self.progress_update.emit("Performing envelope analysis...")
            envelope_analysis = processor.envelope_analysis(signal_data, filter_band=(1000, 8000))
            
            order_analysis = None
            if self.rpm:
                self.progress_update.emit("Performing order tracking...")
                order_analysis = processor.order_tracking(signal_data, self.rpm)
            
            results = {
                'time_data': time_data,
                'signal_data': signal_data,
                'time_features': time_features,
                'frequency_analysis': freq_analysis,
                'envelope_analysis': envelope_analysis,
                'order_analysis': order_analysis,
                'sampling_rate': self.sampling_rate,
                'rpm': self.rpm
            }
            
            self.progress_update.emit("Analysis complete!")
            self.analysis_completed.emit(results)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            self.analysis_error.emit(error_msg)

class PlotWidget(QWidget):
    """Custom widget for matplotlib plots"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def clear_plot(self):
        """Clear the current plot"""
        self.figure.clear()
        self.canvas.draw()
        
    def plot_time_domain(self, time_data: np.ndarray, signal_data: np.ndarray, title: str = "Time Domain"):
        """Plot time domain signal"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(time_data, signal_data, 'b-', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_frequency_domain(self, frequencies: np.ndarray, magnitude: np.ndarray, title: str = "Frequency Domain"):
        """Plot frequency domain spectrum"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.semilogy(frequencies, magnitude, 'r-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, frequencies[-1])
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_order_spectrum(self, orders: np.ndarray, amplitudes: np.ndarray, title: str = "Order Spectrum"):
        """Plot order tracking spectrum"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(orders, amplitudes, width=0.3, alpha=0.7, color='green')
        ax.set_xlabel('Order')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

class VibrationAnalyzerGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.current_results = None
        self.analysis_worker = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Vibration Analyzer Pro v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        self.create_control_panel(splitter)
        
        # Right panel - Results
        self.create_results_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_control_panel(self, parent):
        """Create left control panel"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        
        # File selection group
        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout(file_group)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        self.browse_button = QPushButton("Browse Files...")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        
        layout.addWidget(file_group)
        
        # Analysis parameters group
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QGridLayout(params_group)
        
        # Sampling rate
        params_layout.addWidget(QLabel("Sampling Rate (Hz):"), 0, 0)
        self.sampling_rate_spin = QSpinBox()
        self.sampling_rate_spin.setRange(1000, 100000)
        self.sampling_rate_spin.setValue(20000)
        params_layout.addWidget(self.sampling_rate_spin, 0, 1)
        
        # RPM (optional)
        params_layout.addWidget(QLabel("RPM (optional):"), 1, 0)
        self.rpm_spin = QDoubleSpinBox()
        self.rpm_spin.setRange(0, 10000)
        self.rpm_spin.setValue(1800)
        self.rpm_spin.setSpecialValueText("Not specified")
        params_layout.addWidget(self.rpm_spin, 1, 1)
        
        # Window function
        params_layout.addWidget(QLabel("Window Function:"), 2, 0)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman", "none"])
        params_layout.addWidget(self.window_combo, 2, 1)
        
        layout.addWidget(params_group)
        
        # Analysis button
        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.analyze_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        layout.addWidget(self.analyze_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Progress text
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        # Generate test data button
        self.test_data_button = QPushButton("Generate Test Data")
        self.test_data_button.clicked.connect(self.generate_test_data)
        layout.addWidget(self.test_data_button)
        
        # Export results button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        parent.addWidget(control_widget)
        
    def create_results_panel(self, parent):
        """Create right results panel"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Time domain tab
        self.time_plot = PlotWidget()
        self.tab_widget.addTab(self.time_plot, "Time Domain")
        
        # Frequency domain tab
        self.freq_plot = PlotWidget()
        self.tab_widget.addTab(self.freq_plot, "Frequency Domain")
        
        # Envelope analysis tab
        self.envelope_plot = PlotWidget()
        self.tab_widget.addTab(self.envelope_plot, "Envelope Analysis")
        
        # Order tracking tab
        self.order_plot = PlotWidget()
        self.tab_widget.addTab(self.order_plot, "Order Tracking")
        
        # Results summary tab
        self.create_summary_tab()
        
        parent.addWidget(results_widget)
        
    def create_summary_tab(self):
        """Create summary results tab"""
        summary_widget = QWidget()
        layout = QVBoxLayout(summary_widget)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)
        
        self.tab_widget.addTab(summary_widget, "Summary")
        
    def browse_file(self):
        """Browse for data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vibration Data File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.file_path = file_path
            self.analyze_button.setEnabled(True)
            
    def generate_test_data(self):
        """Generate test data files"""
        try:
            from src.data.mock_generator import generate_test_dataset
            generate_test_dataset()
            QMessageBox.information(self, "Success", "Test data generated successfully in 'test_data' directory!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate test data: {str(e)}")
            
    def run_analysis(self):
        """Run vibration analysis"""
        if not hasattr(self, 'file_path'):
            QMessageBox.warning(self, "Warning", "Please select a data file first.")
            return
            
        # Get parameters
        sampling_rate = self.sampling_rate_spin.value()
        rpm = self.rpm_spin.value() if self.rpm_spin.value() > 0 else None
        
        # Disable controls during analysis
        self.analyze_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start analysis in worker thread
        self.analysis_worker = AnalysisWorker(self.file_path, sampling_rate, rpm)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.analysis_error.connect(self.on_analysis_error)
        self.analysis_worker.progress_update.connect(self.on_progress_update)
        self.analysis_worker.start()
        
    def on_progress_update(self, message: str):
        """Update progress display"""
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
        
    def on_analysis_completed(self, results: dict):
        """Handle completed analysis"""
        self.current_results = results
        
        # Update plots
        self.update_time_domain_plot(results)
        self.update_frequency_domain_plot(results)
        self.update_envelope_plot(results)
        if results['order_analysis']:
            self.update_order_plot(results)
        self.update_summary_table(results)
        
        # Re-enable controls
        self.analyze_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        self.statusBar().showMessage("Analysis completed successfully")
        
    def on_analysis_error(self, error_message: str):
        """Handle analysis error"""
        QMessageBox.critical(self, "Analysis Error", error_message)
        
        # Re-enable controls
        self.analyze_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        self.statusBar().showMessage("Analysis failed")
        
    def update_time_domain_plot(self, results: dict):
        """Update time domain plot"""
        self.time_plot.plot_time_domain(
            results['time_data'],
            results['signal_data'],
            "Time Domain Signal"
        )
        
    def update_frequency_domain_plot(self, results: dict):
        """Update frequency domain plot"""
        freq_analysis = results['frequency_analysis']
        self.freq_plot.plot_frequency_domain(
            freq_analysis['frequencies'],
            freq_analysis['magnitude'],
            "Frequency Spectrum"
        )
        
    def update_envelope_plot(self, results: dict):
        """Update envelope analysis plot"""
        envelope_analysis = results['envelope_analysis']
        envelope_spectrum = envelope_analysis['envelope_spectrum']
        self.envelope_plot.plot_frequency_domain(
            envelope_spectrum['frequencies'],
            envelope_spectrum['magnitude'],
            "Envelope Spectrum (1-8 kHz filtered)"
        )
        
    def update_order_plot(self, results: dict):
        """Update order tracking plot"""
        order_analysis = results['order_analysis']
        self.order_plot.plot_order_spectrum(
            order_analysis['orders'],
            order_analysis['order_amplitudes'],
            f"Order Spectrum (RPM: {results['rpm']})"
        )
        
    def update_summary_table(self, results: dict):
        """Update summary results table"""
        # Prepare summary data
        summary_data = []
        
        # Time domain features
        time_features = results['time_features']
        summary_data.extend([
            ("Signal Length", f"{len(results['signal_data'])} samples"),
            ("Duration", f"{len(results['signal_data'])/results['sampling_rate']:.2f} s"),
            ("RMS", f"{time_features['rms']:.4f}"),
            ("Peak", f"{time_features['peak']:.4f}"),
            ("Crest Factor", f"{time_features['crest_factor']:.2f}"),
            ("Kurtosis", f"{time_features['kurtosis']:.2f}"),
        ])
        
        # Frequency domain features
        freq_features = results['frequency_analysis']['features']
        summary_data.extend([
            ("Peak Frequency", f"{freq_features['peak_frequency']:.1f} Hz"),
            ("Peak Amplitude", f"{freq_features['peak_amplitude']:.4f}"),
            ("Spectral Centroid", f"{freq_features['spectral_centroid']:.1f} Hz"),
        ])
        
        # Order tracking features (if available)
        if results['order_analysis']:
            order_features = results['order_analysis']['features']
            summary_data.extend([
                ("Shaft Frequency", f"{results['order_analysis']['shaft_frequency']:.2f} Hz"),
                ("1X Amplitude", f"{order_features['order_1x_amplitude']:.4f}"),
                ("2X Amplitude", f"{order_features['order_2x_amplitude']:.4f}"),
                ("Dominant Order", f"{order_features['dominant_order']:.1f}"),
            ])
        
        # Update table
        self.results_table.setRowCount(len(summary_data))
        for i, (param, value) in enumerate(summary_data):
            self.results_table.setItem(i, 0, QTableWidgetItem(param))
            self.results_table.setItem(i, 1, QTableWidgetItem(value))
            
    def export_results(self):
        """Export analysis results"""
        if not self.current_results:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "analysis_results.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.save_results_to_csv(file_path)
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
                
    def save_results_to_csv(self, filepath: str):
        """Save results to CSV file"""
        import pandas as pd
        
        results = self.current_results
        
        # Create summary dataframe
        summary_data = {
            'Parameter': [],
            'Value': []
        }
        
        # Add all features to summary
        for i in range(self.results_table.rowCount()):
            param_item = self.results_table.item(i, 0)
            value_item = self.results_table.item(i, 1)
            if param_item and value_item:
                summary_data['Parameter'].append(param_item.text())
                summary_data['Value'].append(value_item.text())
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filepath, index=False)

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = VibrationAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()