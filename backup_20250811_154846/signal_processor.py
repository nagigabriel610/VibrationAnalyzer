"""
Core Signal Processing Engine for Vibration Analysis
Implements FFT, time-domain, envelope, and order tracking analysis
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert, butter, sosfilt
from typing import Dict, Tuple, Optional, List
import warnings

class VibrationProcessor:
    """Main class for vibration signal processing and analysis"""
    
    def __init__(self, sampling_rate: float):
        """
        Initialize the processor
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def load_csv_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load vibration data from CSV file with robust error checking
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Tuple of (time_array, amplitude_array)
        """
        try:
            # Try to read CSV, handling comments and different formats
            df = pd.read_csv(filepath, comment='#')
            
            # Auto-detect time and amplitude columns (improved logic)
            time_col = None
            amp_col = None
            
            # First pass: look for exact or close matches
            for col in df.columns:
                col_lower = col.lower()
                # Time column detection (be more specific)
                if any(x in col_lower for x in ['time_s', 'time', 'timestamp']) and 'amplitude' not in col_lower:
                    time_col = col
                # Amplitude column detection  
                elif any(x in col_lower for x in ['amplitude', 'accel', 'vibration', 'signal']):
                    amp_col = col
            
            # Second pass: fallback to single character matches only if no better match
            if time_col is None:
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower.startswith('t') and 'amplitude' not in col_lower:
                        time_col = col
                        break
            
            if time_col is None:
                # If no time column found, create one
                time_data = np.arange(len(df)) / self.fs
                print("Warning: No time column found, creating time array based on sampling rate")
            else:
                time_data = df[time_col].values
                
            if amp_col is None:
                # Use first numeric column as amplitude
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    amp_col = numeric_cols[0] if time_col != numeric_cols[0] else numeric_cols[1]
                    amp_data = df[amp_col].values
                else:
                    raise ValueError("No numeric columns found for amplitude data")
            else:
                amp_data = df[amp_col].values
            
            # CRITICAL FIX: Check for flipped time array
            if len(time_data) > 1:
                time_diff = np.diff(time_data)
                
                if np.all(time_diff < 0):
                    print("WARNING: Time array is decreasing - automatically flipping data")
                    time_data = np.flip(time_data)
                    amp_data = np.flip(amp_data)
                    # Adjust time to start from 0
                    time_data = time_data - time_data[0]
                    
                elif not np.all(time_diff > 0):
                    print("WARNING: Time array is not monotonically increasing")
                    # Check if it's mostly increasing but has some issues
                    increasing_ratio = np.sum(time_diff > 0) / len(time_diff)
                    if increasing_ratio < 0.9:
                        raise ValueError("Time array is not monotonic and cannot be automatically fixed")
            
            # Verify data quality
            if np.any(np.isnan(time_data)) or np.any(np.isnan(amp_data)):
                raise ValueError("Data contains NaN values")
            
            if np.any(np.isinf(time_data)) or np.any(np.isinf(amp_data)):
                raise ValueError("Data contains infinite values")
                
            print(f"Loaded data: {len(time_data)} samples, {time_data[-1]-time_data[0]:.3f}s duration")
            
            return time_data, amp_data
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    def time_domain_analysis(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Perform time-domain statistical analysis
        
        Args:
            signal_data: Input signal array
            
        Returns:
            Dictionary containing time-domain features
        """
        # Remove DC component
        signal_ac = signal_data - np.mean(signal_data)
        
        # Basic statistical measures
        results = {
            'mean': np.mean(signal_data),
            'rms': np.sqrt(np.mean(signal_ac**2)),
            'std': np.std(signal_data),
            'variance': np.var(signal_data),
            'peak': np.max(np.abs(signal_data)),
            'peak_to_peak': np.max(signal_data) - np.min(signal_data),
            'skewness': self._calculate_skewness(signal_ac),
            'kurtosis': self._calculate_kurtosis(signal_ac),
            'crest_factor': np.max(np.abs(signal_ac)) / np.sqrt(np.mean(signal_ac**2)),
            'clearance_factor': np.max(np.abs(signal_ac)) / np.mean(np.sqrt(np.abs(signal_ac)))**2,
            'shape_factor': np.sqrt(np.mean(signal_ac**2)) / np.mean(np.abs(signal_ac)),
            'impulse_factor': np.max(np.abs(signal_ac)) / np.mean(np.abs(signal_ac))
        }
        
        return results
    
    def frequency_domain_analysis(self, signal_data: np.ndarray, window: str = 'hann') -> Dict:
        """
        Perform FFT-based frequency domain analysis
        
        Args:
            signal_data: Input signal array
            window: Windowing function ('hann', 'hamming', 'blackman', 'none')
            
        Returns:
            Dictionary containing frequency domain data and features
        """
        # Remove DC component
        signal_ac = signal_data - np.mean(signal_data)
        
        # Apply windowing
        if window != 'none':
            if window == 'hann':
                window_func = np.hanning(len(signal_ac))
            elif window == 'hamming':
                window_func = np.hamming(len(signal_ac))
            elif window == 'blackman':
                window_func = np.blackman(len(signal_ac))
            else:
                window_func = np.ones(len(signal_ac))
                
            signal_windowed = signal_ac * window_func
            # Compensate for window power loss
            window_correction = np.sqrt(np.mean(window_func**2))
            signal_windowed = signal_windowed / window_correction
        else:
            signal_windowed = signal_ac
        
        # Compute FFT
        fft_result = np.fft.fft(signal_windowed)
        n_samples = len(signal_windowed)
        
        # Single-sided spectrum
        single_sided = fft_result[:n_samples//2]
        
        # Frequency array
        frequencies = np.fft.fftfreq(n_samples, 1/self.fs)[:n_samples//2]
        
        # Magnitude spectrum (convert to single-sided)
        magnitude = np.abs(single_sided) * 2 / n_samples
        magnitude[0] = magnitude[0] / 2  # DC component shouldn't be doubled
        
        # Power spectral density
        psd = magnitude**2
        
        # Phase spectrum
        phase = np.angle(single_sided)
        
        # Calculate frequency domain features
        features = self._calculate_frequency_features(frequencies, magnitude, psd)
        
        return {
            'frequencies': frequencies,
            'magnitude': magnitude,
            'psd': psd,
            'phase': phase,
            'features': features
        }
    
    def envelope_analysis(self, signal_data: np.ndarray, 
                         filter_band: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Perform envelope analysis using Hilbert transform
        
        Args:
            signal_data: Input signal array
            filter_band: Optional band-pass filter (low_freq, high_freq) in Hz
            
        Returns:
            Dictionary containing envelope spectrum and features
        """
        # Remove DC component
        signal_ac = signal_data - np.mean(signal_data)
        
        # Apply band-pass filter if specified
        if filter_band:
            low_freq, high_freq = filter_band
            if low_freq >= high_freq or high_freq > self.nyquist:
                raise ValueError("Invalid filter band specification")
                
            # Design band-pass filter
            sos = butter(4, [low_freq, high_freq], btype='band', fs=self.fs, output='sos')
            signal_filtered = sosfilt(sos, signal_ac)
        else:
            signal_filtered = signal_ac
        
        # Calculate envelope using Hilbert transform
        analytic_signal = hilbert(signal_filtered)
        envelope = np.abs(analytic_signal)
        
        # Remove DC from envelope
        envelope_ac = envelope - np.mean(envelope)
        
        # Calculate envelope spectrum
        envelope_fft = self.frequency_domain_analysis(envelope_ac)
        
        return {
            'envelope': envelope,
            'envelope_ac': envelope_ac,
            'envelope_spectrum': envelope_fft,
            'filter_band': filter_band
        }
    
    def order_tracking(self, signal_data: np.ndarray, rpm: float, 
                      max_order: int = 20) -> Dict:
        """
        Perform order tracking analysis
        
        Args:
            signal_data: Input signal array
            rpm: Rotational speed in RPM
            max_order: Maximum order to analyze
            
        Returns:
            Dictionary containing order spectrum and features
        """
        # Calculate shaft frequency
        shaft_freq = rpm / 60.0
        
        # Remove DC component
        signal_ac = signal_data - np.mean(signal_data)
        
        # Get frequency spectrum
        freq_analysis = self.frequency_domain_analysis(signal_ac)
        frequencies = freq_analysis['frequencies']
        magnitude = freq_analysis['magnitude']
        
        # Calculate orders and corresponding amplitudes
        orders = np.arange(0.5, max_order + 0.5, 0.5)  # Include half orders
        order_amplitudes = []
        order_frequencies = []
        
        for order in orders:
            target_freq = order * shaft_freq
            order_frequencies.append(target_freq)
            
            if target_freq > self.nyquist:
                order_amplitudes.append(0)
                continue
                
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(frequencies - target_freq))
            
            # Get amplitude in a small band around the target frequency
            bandwidth = shaft_freq * 0.1  # 10% of shaft frequency as bandwidth
            freq_mask = np.abs(frequencies - target_freq) <= bandwidth
            
            if np.any(freq_mask):
                order_amplitude = np.max(magnitude[freq_mask])
            else:
                order_amplitude = magnitude[freq_idx]
                
            order_amplitudes.append(order_amplitude)
        
        order_amplitudes = np.array(order_amplitudes)
        order_frequencies = np.array(order_frequencies)
        
        # Calculate order tracking features
        features = {
            'total_order_energy': np.sum(order_amplitudes**2),
            'dominant_order': orders[np.argmax(order_amplitudes)],
            'dominant_order_amplitude': np.max(order_amplitudes),
            'order_1x_amplitude': order_amplitudes[orders == 1.0][0] if 1.0 in orders else 0,
            'order_2x_amplitude': order_amplitudes[orders == 2.0][0] if 2.0 in orders else 0,
            'order_3x_amplitude': order_amplitudes[orders == 3.0][0] if 3.0 in orders else 0,
        }
        
        return {
            'orders': orders,
            'order_amplitudes': order_amplitudes,
            'order_frequencies': order_frequencies,
            'shaft_frequency': shaft_freq,
            'features': features
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness (third moment)"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis (fourth moment)"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3  # Excess kurtosis
    
    def _calculate_frequency_features(self, frequencies: np.ndarray, 
                                    magnitude: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """Calculate frequency domain features"""
        # Frequency resolution
        df = frequencies[1] - frequencies[0]
        
        # Spectral moments
        m0 = np.sum(psd) * df  # Total power
        m1 = np.sum(frequencies * psd) * df
        m2 = np.sum(frequencies**2 * psd) * df
        m4 = np.sum(frequencies**4 * psd) * df
        
        # Avoid division by zero
        if m0 == 0:
            return {
                'peak_frequency': 0,
                'peak_amplitude': 0,
                'spectral_centroid': 0,
                'spectral_spread': 0,
                'spectral_skewness': 0,
                'spectral_kurtosis': 0,
                'spectral_energy': 0
            }
        
        # Spectral features
        peak_idx = np.argmax(magnitude)
        
        features = {
            'peak_frequency': frequencies[peak_idx],
            'peak_amplitude': magnitude[peak_idx],
            'spectral_centroid': m1 / m0,
            'spectral_spread': np.sqrt(m2 / m0 - (m1 / m0)**2),
            'spectral_energy': m0,
            'spectral_rms_freq': np.sqrt(m2 / m0),
            'frequency_variance': m2 / m0 - (m1 / m0)**2
        }
        
        # Higher order moments (with safety checks)
        if m2 > 0:
            m3 = np.sum((frequencies - features['spectral_centroid'])**3 * psd) * df
            features['spectral_skewness'] = m3 / (features['spectral_spread']**3 * m0)
            
            # Spectral kurtosis approximation
            features['spectral_kurtosis'] = m4 / (m2**2) - 3
        else:
            features['spectral_skewness'] = 0
            features['spectral_kurtosis'] = 0
        
        return features

# Example usage and testing functions
def analyze_signal_file(filepath: str, sampling_rate: float = 20000, rpm: Optional[float] = None):
    """
    Complete analysis of a signal file
    
    Args:
        filepath: Path to CSV file
        sampling_rate: Sampling frequency in Hz
        rpm: Rotational speed in RPM (required for order tracking)
    
    Returns:
        Complete analysis results
    """
    processor = VibrationProcessor(sampling_rate)
    
    # Load data
    time_data, signal_data = processor.load_csv_data(filepath)
    
    print(f"Loaded signal: {len(signal_data)} samples, {len(signal_data)/sampling_rate:.2f} seconds")
    
    # Time domain analysis
    time_features = processor.time_domain_analysis(signal_data)
    print(f"RMS: {time_features['rms']:.4f}, Peak: {time_features['peak']:.4f}")
    
    # Frequency domain analysis
    freq_analysis = processor.frequency_domain_analysis(signal_data)
    freq_features = freq_analysis['features']
    print(f"Peak Frequency: {freq_features['peak_frequency']:.2f} Hz")
    
    # Envelope analysis (high frequency band 1-8 kHz typical for bearings)
    envelope_analysis = processor.envelope_analysis(signal_data, filter_band=(1000, 8000))
    
    # Order tracking (if RPM provided)
    order_analysis = None
    if rpm:
        order_analysis = processor.order_tracking(signal_data, rpm)
        print(f"Shaft Frequency: {order_analysis['shaft_frequency']:.2f} Hz")
        print(f"1X Amplitude: {order_analysis['features']['order_1x_amplitude']:.4f}")
    
    return {
        'time_features': time_features,
        'frequency_analysis': freq_analysis,
        'envelope_analysis': envelope_analysis,
        'order_analysis': order_analysis
    }

if __name__ == "__main__":
    # Test with generated data
    print("Testing signal processor...")
    
    # This would be called after generating test data
    test_file = "test_data/bearing_healthy_1800rpm_01.csv"
    try:
        results = analyze_signal_file(test_file, sampling_rate=20000, rpm=1800)
        print("Signal processing test completed successfully!")
    except FileNotFoundError:
        print("Test data not found. Run mock_generator.py first to create test data.")