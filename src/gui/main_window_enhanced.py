"""
Enhanced Main GUI Window for Vibration Analyzer with Interactive Features
Professional interface with frequency range controls and interactive plots
"""

import sys
import os
import traceback
from typing import Optional, Tuple

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QLabel, QFileDialog, QTextEdit, 
                           QTabWidget, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
                           QComboBox, QProgressBar, QMessageBox, QTableWidget, 
                           QTableWidgetItem, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from scipy.signal import find_peaks
import numpy as np

# Import our analysis modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.analysis.signal_processor import VibrationProcessor

class InteractivePlotWidget(QWidget):
    """Enhanced plot widget with interactive features for frequency analysis"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Store current data for interactive features
        self.current_frequencies = None
        self.current_magnitude = None
        self.current_amplitude_scale = "Linear"
        self.zoom_history = []
        
        # Setup layout
        self.setup_layout()
        
        # Connect interactive events
        self.setup_interactions()
        
    def setup_layout(self):
        """Setup the widget layout with toolbar and controls"""
        layout = QVBoxLayout()
        
        # Create matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        
        # Add plot canvas
        layout.addWidget(self.canvas)
        
        # Create interactive controls
        controls_layout = QHBoxLayout()
        
        # Zoom controls group
        zoom_group = QGroupBox("Zoom Controls")
        zoom_layout = QHBoxLayout(zoom_group)
        
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.auto_scale_btn = QPushButton("Auto Scale")
        
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        self.auto_scale_btn.clicked.connect(self.auto_scale)
        
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.reset_zoom_btn)
        zoom_layout.addWidget(self.auto_scale_btn)
        
        controls_layout.addWidget(zoom_group)
        
        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QHBoxLayout(display_group)
        
        self.show_peaks_cb = QCheckBox("Show Peaks")
        self.show_grid_cb = QCheckBox("Show Grid")
        self.show_cursor_cb = QCheckBox("Show Cursor Info")
        
        self.show_peaks_cb.setChecked(False)
        self.show_grid_cb.setChecked(True)
        self.show_cursor_cb.setChecked(True)
        
        self.show_peaks_cb.toggled.connect(self.update_plot_features)
        self.show_grid_cb.toggled.connect(self.update_plot_features)
        self.show_cursor_cb.toggled.connect(self.toggle_cursor)
        
        display_layout.addWidget(self.show_peaks_cb)
        display_layout.addWidget(self.show_grid_cb)
        display_layout.addWidget(self.show_cursor_cb)
        
        controls_layout.addWidget(display_group)
        
        # Frequency info label
        self.freq_info_label = QLabel("Frequency: -- Hz, Amplitude: --")
        controls_layout.addWidget(self.freq_info_label)
        
        layout.addLayout(controls_layout)
        self.setLayout(layout)
        
    def setup_interactions(self):
        """Setup interactive event handlers"""
        # Mouse motion for cursor tracking
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Mouse click for zoom to region
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        # Key press for shortcuts
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Span selector for region zoom (will be created when plotting)
        self.span_selector = None
        
    def clear_plot(self):
        """Clear the current plot"""
        self.figure.clear()
        self.canvas.draw()
        self.current_frequencies = None
        self.current_magnitude = None
        
    def plot_frequency_domain_interactive(self, frequencies, magnitude, amplitude_scale="Linear", 
                                        title="Interactive Frequency Spectrum"):
        """Plot interactive frequency domain with all features"""
        # Store data for interactive features
        self.current_frequencies = frequencies.copy()
        self.current_magnitude = magnitude.copy()
        self.current_amplitude_scale = amplitude_scale
        
        # Clear previous plot
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Plot based on amplitude scale
        if amplitude_scale == "Logarithmic (dB)":
            # dB mode: plot dB values on linear axis
            self.ax.plot(frequencies, magnitude, 'b-', linewidth=1, label='Magnitude (dB)')
            self.ax.set_ylabel('Magnitude (dB)')
            self.ax.set_yscale('linear')
        else:
            # Linear mode: plot magnitude values on linear axis
            self.ax.plot(frequencies, magnitude, 'b-', linewidth=1, label='Magnitude')
            self.ax.set_ylabel('Magnitude')
            self.ax.set_yscale('linear')
        
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_title(title)
        
        # Apply current display options
        self.update_plot_features()
        
        # Setup span selector for zoom
        self.setup_span_selector()
        
        # Auto scale initially
        self.auto_scale()
        
        # Draw
        self.figure.tight_layout()
        self.canvas.draw()
        
    def setup_span_selector(self):
        """Setup span selector for region zooming"""
        if hasattr(self, 'ax') and self.ax and self.current_frequencies is not None:
            self.span_selector = SpanSelector(
                self.ax, self.on_span_select, 'horizontal',
                useblit=True, props=dict(alpha=0.3, facecolor='red')
            )
            
    def on_span_select(self, xmin, xmax):
        """Handle span selection for zooming"""
        if xmin == xmax:
            return
            
        # Store current zoom in history
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        self.zoom_history.append((current_xlim, current_ylim))
        
        # Apply zoom
        self.ax.set_xlim(xmin, xmax)
        
        # Auto-scale y-axis for the selected range
        freq_mask = (self.current_frequencies >= xmin) & (self.current_frequencies <= xmax)
        if np.any(freq_mask):
            y_data = self.current_magnitude[freq_mask]
            if len(y_data) > 0:
                if self.current_amplitude_scale == "Logarithmic (dB)":
                    y_margin = (np.max(y_data) - np.min(y_data)) * 0.1
                    self.ax.set_ylim(np.min(y_data) - y_margin, np.max(y_data) + y_margin)
                else:
                    self.ax.set_ylim(np.min(y_data) * 0.8, np.max(y_data) * 1.2)
        
        self.canvas.draw()
        
    def on_mouse_move(self, event):
        """Handle mouse movement for cursor tracking"""
        if (hasattr(self, 'ax') and event.inaxes == self.ax and self.show_cursor_cb.isChecked() 
            and self.current_frequencies is not None):
            
            if event.xdata is not None and event.ydata is not None:
                # Find closest frequency point
                freq_idx = np.argmin(np.abs(self.current_frequencies - event.xdata))
                closest_freq = self.current_frequencies[freq_idx]
                closest_amp = self.current_magnitude[freq_idx]
                
                # Update info label
                if self.current_amplitude_scale == "Logarithmic (dB)":
                    self.freq_info_label.setText(
                        f"Frequency: {closest_freq:.1f} Hz, Amplitude: {closest_amp:.2f} dB"
                    )
                else:
                    self.freq_info_label.setText(
                        f"Frequency: {closest_freq:.1f} Hz, Amplitude: {closest_amp:.4f}"
                    )
                    
    def on_mouse_click(self, event):
        """Handle mouse clicks for quick zoom"""
        if hasattr(self, 'ax') and event.inaxes == self.ax and event.dblclick:
            # Double-click to zoom in around clicked point
            x_center = event.xdata
            current_xlim = self.ax.get_xlim()
            x_range = current_xlim[1] - current_xlim[0]
            
            # Store current zoom in history
            current_ylim = self.ax.get_ylim()
            self.zoom_history.append((current_xlim, current_ylim))
            
            # Zoom in by factor of 2 around clicked point
            new_range = x_range / 2
            self.ax.set_xlim(x_center - new_range/2, x_center + new_range/2)
            self.canvas.draw()
            
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'r':
            self.reset_zoom()
        elif event.key == 'a':
            self.auto_scale()
        elif event.key == 'g':
            self.show_grid_cb.setChecked(not self.show_grid_cb.isChecked())
        elif event.key == 'p':
            self.show_peaks_cb.setChecked(not self.show_peaks_cb.isChecked())
            
    def update_plot_features(self):
        """Update plot features based on checkboxes"""
        if not hasattr(self, 'ax') or self.ax is None:
            return
            
        # Update grid
        self.ax.grid(self.show_grid_cb.isChecked(), alpha=0.3)
        
        # Update peaks
        if self.show_peaks_cb.isChecked() and self.current_magnitude is not None:
            self.show_peaks()
        else:
            self.hide_peaks()
            
        self.canvas.draw()
        
    def show_peaks(self):
        """Find and highlight peaks in the spectrum"""
        if self.current_magnitude is None:
            return
            
        # Find peaks (adjust parameters based on amplitude scale)
        if self.current_amplitude_scale == "Logarithmic (dB)":
            # For dB scale, use absolute threshold
            peaks, properties = find_peaks(self.current_magnitude, 
                                         height=-40,  # -40 dB threshold
                                         distance=10)  # Minimum distance between peaks
        else:
            # For linear scale, use relative threshold
            threshold = np.max(self.current_magnitude) * 0.1  # 10% of max
            peaks, properties = find_peaks(self.current_magnitude, 
                                         height=threshold,
                                         distance=10)
        
        # Remove previous peak markers
        self.hide_peaks()
        
        # Add new peak markers
        if len(peaks) > 0:
            peak_freqs = self.current_frequencies[peaks]
            peak_amps = self.current_magnitude[peaks]
            
            # Plot peak markers
            self.peak_markers = self.ax.plot(peak_freqs, peak_amps, 'ro', 
                                           markersize=8, alpha=0.7, 
                                           label=f'{len(peaks)} peaks found')[0]
            
            # Add annotations for top 5 peaks
            if len(peaks) > 0:
                # Sort peaks by amplitude
                if self.current_amplitude_scale == "Logarithmic (dB)":
                    peak_indices = np.argsort(peak_amps)[-5:]  # Top 5 for dB
                else:
                    peak_indices = np.argsort(peak_amps)[-5:]  # Top 5 for linear
                
                for i in peak_indices:
                    if i < len(peak_freqs):
                        freq = peak_freqs[i]
                        amp = peak_amps[i]
                        if self.current_amplitude_scale == "Logarithmic (dB)":
                            self.ax.annotate(f'{freq:.1f} Hz\n{amp:.1f} dB', 
                                           (freq, amp),
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=8, alpha=0.8)
                        else:
                            self.ax.annotate(f'{freq:.1f} Hz', 
                                           (freq, amp),
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=8, alpha=0.8)
            
            # Update legend
            self.ax.legend()

    def hide_peaks(self):
        """Remove peak markers from plot"""
        if hasattr(self, 'peak_markers') and self.peak_markers:
            try:
                self.peak_markers.remove()
            except:
                pass  # Ignore removal errors
            self.peak_markers = None
            
        # Remove annotations safely
        try:
            # Clear all annotations by recreating the plot data
            if hasattr(self, 'ax') and self.ax and self.current_frequencies is not None:
                # Store current plot data
                current_lines = []
                for line in self.ax.get_lines():
                    if not (hasattr(line, '_label') and 'peaks found' in str(line._label)):
                        current_lines.append((line.get_xdata(), line.get_ydata(), line.get_color(), line.get_linewidth()))
                
                # Clear axis and redraw main plot
                self.ax.clear()
                
                # Redraw the main spectrum
                for xdata, ydata, color, linewidth in current_lines:
                    if len(xdata) > 100:  # This is likely the main spectrum
                        if self.current_amplitude_scale == "Logarithmic (dB)":
                            self.ax.plot(xdata, ydata, color=color, linewidth=linewidth)
                            self.ax.set_ylabel('Magnitude (dB)')
                        else:
                            self.ax.plot(xdata, ydata, color=color, linewidth=linewidth)
                            self.ax.set_ylabel('Magnitude')
                        break
                
                self.ax.set_xlabel('Frequency (Hz)')
                self.ax.grid(self.show_grid_cb.isChecked(), alpha=0.3)
                
        except Exception as e:
            # If all else fails, just ignore the annotation removal
            print(f"Note: Could not remove peak annotations (matplotlib compatibility): {e}")
        
        # Update legend if axis exists
        if hasattr(self, 'ax') and self.ax:
            try:
                self.ax.legend()
            except:
                pass
        
    def zoom_in(self):
        """Zoom in by factor of 2"""
        if not hasattr(self, 'ax'):
            return
            
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Store in history
        self.zoom_history.append((current_xlim, current_ylim))
        
        # Calculate new limits
        x_center = (current_xlim[0] + current_xlim[1]) / 2
        x_range = (current_xlim[1] - current_xlim[0]) / 2
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.canvas.draw()
        
    def zoom_out(self):
        """Zoom out by factor of 2"""
        if not hasattr(self, 'ax'):
            return
            
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Store in history
        self.zoom_history.append((current_xlim, current_ylim))
        
        # Calculate new limits
        x_center = (current_xlim[0] + current_xlim[1]) / 2
        x_range = (current_xlim[1] - current_xlim[0]) * 2
        
        # Don't zoom out beyond original data range
        if self.current_frequencies is not None:
            max_range = np.max(self.current_frequencies) - np.min(self.current_frequencies)
            x_range = min(x_range, max_range)
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.canvas.draw()
        
    def reset_zoom(self):
        """Reset zoom to show full spectrum"""
        if self.current_frequencies is not None:
            self.ax.set_xlim(np.min(self.current_frequencies), 
                           np.max(self.current_frequencies))
            self.auto_scale_y()
            self.canvas.draw()
            
    def auto_scale(self):
        """Auto-scale both axes"""
        if self.current_frequencies is not None:
            self.ax.set_xlim(np.min(self.current_frequencies), 
                           np.max(self.current_frequencies))
            self.auto_scale_y()
            self.canvas.draw()
            
    def auto_scale_y(self):
        """Auto-scale y-axis based on current x-range"""
        if self.current_frequencies is None or self.current_magnitude is None:
            return
            
        # Get current x-limits
        x_min, x_max = self.ax.get_xlim()
        
        # Find data within current x-range
        freq_mask = ((self.current_frequencies >= x_min) & 
                    (self.current_frequencies <= x_max))
        
        if np.any(freq_mask):
            y_data = self.current_magnitude[freq_mask]
            if len(y_data) > 0:
                if self.current_amplitude_scale == "Logarithmic (dB)":
                    y_margin = (np.max(y_data) - np.min(y_data)) * 0.1
                    self.ax.set_ylim(np.min(y_data) - y_margin, 
                                   np.max(y_data) + y_margin)
                else:
                    self.ax.set_ylim(np.min(y_data) * 0.8, 
                                   np.max(y_data) * 1.2)
                    
    def toggle_cursor(self):
        """Toggle cursor info display"""
        if not self.show_cursor_cb.isChecked():
            self.freq_info_label.setText("Frequency: -- Hz, Amplitude: --")

    # Backwards compatibility methods
    def plot_time_domain(self, time_data, signal_data, title="Time Domain"):
        """Plot time domain signal (non-interactive)"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(time_data, signal_data, 'b-', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_frequency_domain(self, frequencies, magnitude, title="Frequency Domain"):
        """Plot frequency domain using interactive features"""
        self.plot_frequency_domain_interactive(frequencies, magnitude, "Linear", title)
        
    def plot_order_spectrum(self, orders, amplitudes, title="Order Spectrum"):
        """Plot order tracking spectrum (non-interactive)"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(orders, amplitudes, width=0.3, alpha=0.7, color='green')
        ax.set_xlabel('Order')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

class EnhancedAnalysisWorker(QThread):
    """Enhanced worker thread with custom frequency range support"""
    
    analysis_completed = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, filepath: str, sampling_rate: float, rpm: Optional[float] = None, 
                 freq_range: Optional[Tuple[float, float]] = None, 
                 amplitude_scale: str = "Linear"):
        super().__init__()
        self.filepath = filepath
        self.sampling_rate = sampling_rate
        self.rpm = rpm
        self.freq_range = freq_range
        self.amplitude_scale = amplitude_scale
        
    def run(self):
        """Run enhanced analysis with custom frequency ranges"""
        try:
            processor = VibrationProcessor(self.sampling_rate)
            
            self.progress_update.emit("Loading data...")
            time_data, signal_data = processor.load_csv_data(self.filepath)
            
            self.progress_update.emit("Performing time-domain analysis...")
            time_features = processor.time_domain_analysis(signal_data)
            
            self.progress_update.emit("Performing frequency-domain analysis...")
            freq_analysis = processor.frequency_domain_analysis(signal_data)
            
            # Apply frequency range filtering if specified
            if self.freq_range:
                freq_analysis = self.apply_frequency_range(freq_analysis, self.freq_range)
            
            # Apply amplitude scaling
            if self.amplitude_scale == "Logarithmic (dB)":
                freq_analysis = self.apply_log_scale(freq_analysis)
            
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
                'rpm': self.rpm,
                'freq_range': self.freq_range,
                'amplitude_scale': self.amplitude_scale
            }
            
            self.progress_update.emit("Analysis complete!")
            self.analysis_completed.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            self.analysis_error.emit(error_msg)
    
    def apply_frequency_range(self, freq_analysis, freq_range):
        """Apply custom frequency range to analysis results"""
        frequencies = freq_analysis['frequencies']
        min_freq, max_freq = freq_range
        
        # Find indices within the specified range
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        
        # Filter all frequency domain data
        filtered_analysis = {
            'frequencies': frequencies[freq_mask],
            'magnitude': freq_analysis['magnitude'][freq_mask],
            'psd': freq_analysis['psd'][freq_mask],
            'phase': freq_analysis['phase'][freq_mask],
            'features': freq_analysis['features']  # Keep original features
        }
        
        return filtered_analysis
    
    def apply_log_scale(self, freq_analysis):
        """Convert magnitude to logarithmic scale (dB)"""
        magnitude_linear = freq_analysis['magnitude']
        
        # Convert to dB (20*log10 for amplitude)
        # Add small value to avoid log(0)
        magnitude_db = 20 * np.log10(magnitude_linear + 1e-12)
        
        freq_analysis_log = freq_analysis.copy()
        freq_analysis_log['magnitude'] = magnitude_db
        freq_analysis_log['amplitude_scale'] = 'dB'
        
        return freq_analysis_log

class VibrationAnalyzerGUI(QMainWindow):
    """Enhanced main application window with interactive features"""
    
    def __init__(self):
        super().__init__()
        self.current_results = None
        self.analysis_worker = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize enhanced user interface"""
        self.setWindowTitle("Vibration Analyzer Pro v1.1 - Interactive Analysis")
        self.setGeometry(100, 100, 1600, 1000)  # Larger window for new features
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Enhanced controls
        self.create_enhanced_control_panel(splitter)
        
        # Right panel - Enhanced results
        self.create_enhanced_results_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([450, 1150])  # More space for controls
        
        # Status bar
        self.statusBar().showMessage("Ready - Enhanced Interactive Features Enabled")
        
    def create_enhanced_control_panel(self, parent):
        """Create enhanced left control panel with frequency analysis controls"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        
        # File selection group (unchanged)
        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout(file_group)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        self.browse_button = QPushButton("Browse Files...")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        
        layout.addWidget(file_group)
        
        # Analysis parameters group (enhanced)
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QGridLayout(params_group)
        
        # Sampling rate
        params_layout.addWidget(QLabel("Sampling Rate (Hz):"), 0, 0)
        self.sampling_rate_spin = QSpinBox()
        self.sampling_rate_spin.setRange(1000, 100000)
        self.sampling_rate_spin.setValue(20000)
        self.sampling_rate_spin.valueChanged.connect(self.update_frequency_limits)
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
        
        # NEW: Frequency Analysis Group
        freq_group = QGroupBox("Frequency Analysis Settings")
        freq_layout = QGridLayout(freq_group)
        
        # Frequency range controls
        freq_layout.addWidget(QLabel("Analysis Range:"), 0, 0, 1, 2)
        
        # Auto range checkbox
        self.auto_freq_range = QCheckBox("Auto Range (0 to Nyquist)")
        self.auto_freq_range.setChecked(True)
        self.auto_freq_range.toggled.connect(self.toggle_freq_range)
        freq_layout.addWidget(self.auto_freq_range, 1, 0, 1, 2)
        
        # Manual frequency range
        freq_layout.addWidget(QLabel("Min Freq (Hz):"), 2, 0)
        self.min_freq_spin = QDoubleSpinBox()
        self.min_freq_spin.setRange(0, 50000)
        self.min_freq_spin.setValue(0)
        self.min_freq_spin.setEnabled(False)
        freq_layout.addWidget(self.min_freq_spin, 2, 1)
        
        freq_layout.addWidget(QLabel("Max Freq (Hz):"), 3, 0)
        self.max_freq_spin = QDoubleSpinBox()
        self.max_freq_spin.setRange(1, 50000)
        self.max_freq_spin.setValue(10000)
        self.max_freq_spin.setEnabled(False)
        freq_layout.addWidget(self.max_freq_spin, 3, 1)
        
        # Amplitude scale selection
        freq_layout.addWidget(QLabel("Amplitude Scale:"), 4, 0)
        self.amplitude_scale_combo = QComboBox()
        self.amplitude_scale_combo.addItems(["Linear", "Logarithmic (dB)"])
        self.amplitude_scale_combo.currentTextChanged.connect(self.update_amplitude_scale)
        freq_layout.addWidget(self.amplitude_scale_combo, 4, 1)
        
        layout.addWidget(freq_group)
        
        # Analysis button (enhanced)
        self.analyze_button = QPushButton("🔍 Run Enhanced Analysis")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.analyze_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 12px; 
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { 
                background-color: #45a049; 
            }
        """)
        layout.addWidget(self.analyze_button)
        
        # Progress controls (unchanged)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        # Other buttons (unchanged)
        self.test_data_button = QPushButton("Generate Test Data")
        self.test_data_button.clicked.connect(self.generate_test_data)
        layout.addWidget(self.test_data_button)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        layout.addWidget(self.export_button)
        
        # Add help text
        help_label = QLabel("""
💡 Interactive Features:
• Drag to select zoom region
• Double-click to zoom in
• Use toolbar for pan/zoom
• Press 'r' to reset zoom
• Press 'g' to toggle grid
• Press 'p' to toggle peaks
        """)
        help_label.setStyleSheet("color: #666; font-size: 10px; padding: 10px;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        parent.addWidget(control_widget)
    
    def create_enhanced_results_panel(self, parent):
        """Create enhanced right results panel with interactive plots"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Time domain tab (enhanced plot widget)
        self.time_plot = InteractivePlotWidget()
        self.tab_widget.addTab(self.time_plot, "Time Domain")
        
        # Frequency domain tab (ENHANCED with interactions)
        self.freq_plot = InteractivePlotWidget()
        self.tab_widget.addTab(self.freq_plot, "🔍 Interactive Frequency")
        
        # Envelope analysis tab
        self.envelope_plot = InteractivePlotWidget()
        self.tab_widget.addTab(self.envelope_plot, "Envelope Analysis")
        
        # Order tracking tab
        self.order_plot = InteractivePlotWidget()
        self.tab_widget.addTab(self.order_plot, "Order Tracking")
        
        # Results summary tab
        self.create_summary_tab()
        
        parent.addWidget(results_widget)
        
    def create_summary_tab(self):
        """Create enhanced summary results tab"""
        summary_widget = QWidget()
        layout = QVBoxLayout(summary_widget)
        
        # Analysis info label
        self.analysis_info_label = QLabel("No analysis performed yet")
        self.analysis_info_label.setStyleSheet("font-weight: bold; color: #2c3e50; padding: 10px;")
        layout.addWidget(self.analysis_info_label)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)
        
        self.tab_widget.addTab(summary_widget, "📊 Summary")
    
    def update_frequency_limits(self):
        """Update frequency range limits based on sampling rate"""
        fs = self.sampling_rate_spin.value()
        nyquist = fs / 2
        
        self.max_freq_spin.setMaximum(nyquist)
        if self.auto_freq_range.isChecked():
            self.max_freq_spin.setValue(nyquist)
    
    def toggle_freq_range(self, checked):
        """Toggle between auto and manual frequency range"""
        self.min_freq_spin.setEnabled(not checked)
        self.max_freq_spin.setEnabled(not checked)
        
        if checked:
            # Auto range - update based on sampling rate
            fs = self.sampling_rate_spin.value()
            self.min_freq_spin.setValue(0)
            self.max_freq_spin.setValue(fs / 2)
        
        # If we have current results, update the plots
        if hasattr(self, 'current_results') and self.current_results:
            self.update_frequency_domain_plot(self.current_results)
    
    def update_amplitude_scale(self):
        """Update amplitude scale and refresh plots if data exists"""
        if hasattr(self, 'current_results') and self.current_results:
            self.update_frequency_domain_plot(self.current_results)
    
    def browse_file(self):
        """Browse for data file (unchanged)"""
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
        """Generate test data files (unchanged)"""
        try:
            from src.data.mock_generator import generate_test_dataset
            generate_test_dataset()
            QMessageBox.information(self, "Success", "Test data generated successfully in 'test_data' directory!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate test data: {str(e)}")
            
    def run_analysis(self):
        """Enhanced analysis method with frequency range support"""
        if not hasattr(self, 'file_path'):
            QMessageBox.warning(self, "Warning", "Please select a data file first.")
            return
            
        # Get parameters
        sampling_rate = self.sampling_rate_spin.value()
        rpm = self.rpm_spin.value() if self.rpm_spin.value() > 0 else None
        
        # Get frequency range
        if self.auto_freq_range.isChecked():
            freq_range = None  # Use full range
        else:
            min_freq = self.min_freq_spin.value()
            max_freq = self.max_freq_spin.value()
            
            # Validate frequency range
            if min_freq >= max_freq:
                QMessageBox.warning(self, "Invalid Range", 
                                  "Minimum frequency must be less than maximum frequency.")
                return
            
            if max_freq > sampling_rate / 2:
                QMessageBox.warning(self, "Invalid Range", 
                                  f"Maximum frequency cannot exceed Nyquist frequency ({sampling_rate/2:.0f} Hz).")
                return
                
            freq_range = (min_freq, max_freq)
        
        # Get amplitude scale
        amplitude_scale = self.amplitude_scale_combo.currentText()
        
        # Disable controls during analysis
        self.analyze_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start enhanced analysis worker
        self.analysis_worker = EnhancedAnalysisWorker(
            self.file_path, sampling_rate, rpm, freq_range, amplitude_scale
        )
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.analysis_error.connect(self.on_analysis_error)
        self.analysis_worker.progress_update.connect(self.on_progress_update)
        self.analysis_worker.start()
        
    def on_progress_update(self, message: str):
        """Update progress display (unchanged)"""
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
        
    def on_analysis_completed(self, results: dict):
        """Handle completed enhanced analysis"""
        self.current_results = results
        
        # Update analysis info
        info_text = f"""
Analysis completed with enhanced features:
• File: {os.path.basename(self.file_path)}
• Sampling Rate: {results['sampling_rate']} Hz
• Duration: {len(results['signal_data'])/results['sampling_rate']:.2f} seconds
• Amplitude Scale: {results.get('amplitude_scale', 'Linear')}
        """
        if results.get('freq_range'):
            min_f, max_f = results['freq_range']
            info_text += f"• Frequency Range: {min_f:.1f} - {max_f:.1f} Hz"
        else:
            info_text += f"• Frequency Range: Auto (0 - {results['sampling_rate']/2:.0f} Hz)"
            
        self.analysis_info_label.setText(info_text)
        
        # Update plots with enhanced features
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
        
        self.statusBar().showMessage("Enhanced analysis completed successfully - Try the interactive features!")
        
    def on_analysis_error(self, error_message: str):
        """Handle analysis error (unchanged)"""
        QMessageBox.critical(self, "Analysis Error", error_message)
        
        # Re-enable controls
        self.analyze_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        self.statusBar().showMessage("Analysis failed")
        
    def update_time_domain_plot(self, results: dict):
        """Update time domain plot (unchanged)"""
        self.time_plot.plot_time_domain(
            results['time_data'],
            results['signal_data'],
            "Time Domain Signal"
        )
        
    def update_frequency_domain_plot(self, results: dict):
        """Update frequency domain plot with interactive features"""
        freq_analysis = results['frequency_analysis']
        amplitude_scale = results.get('amplitude_scale', 'Linear')
        
        # Create title with current settings
        title = f"Interactive Frequency Spectrum ({amplitude_scale})"
        if results.get('freq_range'):
            min_f, max_f = results['freq_range']
            title += f" - Range: {min_f:.0f}-{max_f:.0f} Hz"
        
        self.freq_plot.plot_frequency_domain_interactive(
            freq_analysis['frequencies'],
            freq_analysis['magnitude'],
            amplitude_scale,
            title
        )
        
    def update_envelope_plot(self, results: dict):
        """Update envelope analysis plot"""
        envelope_analysis = results['envelope_analysis']
        envelope_spectrum = envelope_analysis['envelope_spectrum']
        self.envelope_plot.plot_frequency_domain_interactive(
            envelope_spectrum['frequencies'],
            envelope_spectrum['magnitude'],
            "Linear",
            "Envelope Spectrum (1-8 kHz filtered)"
        )
        
    def update_order_plot(self, results: dict):
        """Update order tracking plot (unchanged)"""
        order_analysis = results['order_analysis']
        self.order_plot.plot_order_spectrum(
            order_analysis['orders'],
            order_analysis['order_amplitudes'],
            f"Order Spectrum (RPM: {results['rpm']})"
        )
        
    def update_summary_table(self, results: dict):
        """Update enhanced summary results table"""
        # Prepare summary data
        summary_data = []
        
        # File info
        summary_data.extend([
            ("File Name", os.path.basename(self.file_path)),
            ("Signal Length", f"{len(results['signal_data'])} samples"),
            ("Duration", f"{len(results['signal_data'])/results['sampling_rate']:.2f} s"),
            ("Sampling Rate", f"{results['sampling_rate']} Hz"),
        ])
        
        # Frequency analysis settings
        if results.get('freq_range'):
            min_f, max_f = results['freq_range']
            summary_data.append(("Frequency Range", f"{min_f:.1f} - {max_f:.1f} Hz"))
        else:
            summary_data.append(("Frequency Range", f"Auto (0 - {results['sampling_rate']/2:.0f} Hz)"))
            
        summary_data.append(("Amplitude Scale", results.get('amplitude_scale', 'Linear')))
        
        # Time domain features
        time_features = results['time_features']
        summary_data.extend([
            ("RMS", f"{time_features['rms']:.4f}"),
            ("Peak", f"{time_features['peak']:.4f}"),
            ("Crest Factor", f"{time_features['crest_factor']:.2f}"),
            ("Kurtosis", f"{time_features['kurtosis']:.2f}"),
        ])
        
        # Frequency domain features
        freq_features = results['frequency_analysis']['features']
        if results.get('amplitude_scale') == 'Logarithmic (dB)':
            summary_data.extend([
                ("Peak Frequency", f"{freq_features['peak_frequency']:.1f} Hz"),
                ("Peak Amplitude", f"{freq_features['peak_amplitude']:.2f} dB"),
            ])
        else:
            summary_data.extend([
                ("Peak Frequency", f"{freq_features['peak_frequency']:.1f} Hz"),
                ("Peak Amplitude", f"{freq_features['peak_amplitude']:.4f}"),
            ])
            
        summary_data.append(("Spectral Centroid", f"{freq_features['spectral_centroid']:.1f} Hz"))
        
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
        """Export enhanced analysis results"""
        if not self.current_results:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Enhanced Results",
            "enhanced_analysis_results.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.save_results_to_csv(file_path)
                QMessageBox.information(self, "Success", f"Enhanced results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
                
    def save_results_to_csv(self, filepath: str):
        """Save enhanced results to CSV file"""
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
