"""
Mock Data Generator for Radial Ball Bearing Vibration Signals
Generates realistic bearing fault signatures based on industrial standards
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple

class BearingDataGenerator:
    """Generates realistic bearing vibration data with various fault conditions"""
    
    def __init__(self, sampling_rate: int = 20000):
        """
        Initialize the generator
        
        Args:
            sampling_rate: Sampling frequency in Hz (default 20kHz)
        """
        self.fs = sampling_rate
        
        # Standard radial ball bearing parameters (SKF 6205 as example)
        self.bearing_params = {
            'pitch_diameter': 39.04,    # mm
            'ball_diameter': 7.94,      # mm  
            'number_of_balls': 9,
            'contact_angle': 0          # radians for radial bearing
        }
    
    def calculate_bearing_frequencies(self, rpm: float) -> Dict[str, float]:
        """
        Calculate characteristic bearing frequencies
        
        Args:
            rpm: Rotational speed in RPM
            
        Returns:
            Dictionary containing all bearing frequencies
        """
        # Convert RPM to Hz
        shaft_freq = rpm / 60.0
        
        # Bearing geometry ratios
        d = self.bearing_params['ball_diameter']
        D = self.bearing_params['pitch_diameter'] 
        n = self.bearing_params['number_of_balls']
        alpha = self.bearing_params['contact_angle']
        
        bd_ratio = d / D
        
        # Calculate characteristic frequencies
        frequencies = {
            'shaft_frequency': shaft_freq,
            'bpfo': (n / 2) * shaft_freq * (1 - bd_ratio * np.cos(alpha)),  # Ball Pass Frequency Outer
            'bpfi': (n / 2) * shaft_freq * (1 + bd_ratio * np.cos(alpha)),  # Ball Pass Frequency Inner  
            'bsf': (D / (2 * d)) * shaft_freq * (1 - (bd_ratio * np.cos(alpha))**2),  # Ball Spin Frequency
            'ftf': (1 / 2) * shaft_freq * (1 - bd_ratio * np.cos(alpha))   # Fundamental Train Frequency
        }
        
        return frequencies
    
    def generate_healthy_signal(self, rpm: float, duration: float, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate healthy bearing vibration signal
        
        Args:
            rpm: Rotational speed in RPM
            duration: Signal duration in seconds
            noise_level: Relative noise level (0-1)
            
        Returns:
            Tuple of (time_array, signal_array)
        """
        # Create time array - CRITICAL: ensure it's monotonically increasing
        n_samples = int(self.fs * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Verify time array is correct
        assert len(t) == n_samples, "Time array length mismatch"
        assert np.all(np.diff(t) > 0), "Time array is not monotonically increasing"
        
        # Get bearing frequencies
        freqs = self.calculate_bearing_frequencies(rpm)
        shaft_freq = freqs['shaft_frequency']
        
        # Create base signal with shaft frequency and harmonics
        signal = (0.5 * np.sin(2 * np.pi * shaft_freq * t) +           # 1X shaft
                 0.2 * np.sin(2 * np.pi * 2 * shaft_freq * t) +        # 2X shaft  
                 0.1 * np.sin(2 * np.pi * 3 * shaft_freq * t))         # 3X shaft
        
        # Add broadband noise
        noise = noise_level * np.random.normal(0, 1, len(t))
        
        # Add some realistic bearing resonances around 1-5 kHz
        resonance_freq = 2500 + np.random.normal(0, 500)
        resonance = 0.05 * np.sin(2 * np.pi * resonance_freq * t) * np.exp(-t/0.5)
        
        signal_total = signal + noise + resonance
        
        # Final verification
        assert len(t) == len(signal_total), "Time and signal arrays have different lengths"
        assert np.all(np.isfinite(t)), "Time array contains non-finite values"
        assert np.all(np.isfinite(signal_total)), "Signal array contains non-finite values"
        
        return t, signal_total
    
    def generate_fault_signal(self, rpm: float, duration: float, fault_type: str, 
                            fault_severity: float = 0.5, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate bearing signal with specific fault
        
        Args:
            rpm: Rotational speed in RPM
            duration: Signal duration in seconds  
            fault_type: 'inner_race', 'outer_race', 'ball', or 'cage'
            fault_severity: Fault severity (0-1, where 1 is severe)
            noise_level: Relative noise level (0-1)
            
        Returns:
            Tuple of (time_array, signal_array)
        """
        # Start with healthy signal (this ensures proper time array)
        t, healthy_signal = self.generate_healthy_signal(rpm, duration, noise_level)
        
        # Verify we got a proper time array
        assert np.all(np.diff(t) > 0), "Time array from healthy signal is not monotonic"
        
        # Get bearing frequencies
        freqs = self.calculate_bearing_frequencies(rpm)
        
        # Select fault frequency based on fault type
        fault_freq_map = {
            'inner_race': freqs['bpfi'],
            'outer_race': freqs['bpfo'], 
            'ball': freqs['bsf'],
            'cage': freqs['ftf']
        }
        
        if fault_type not in fault_freq_map:
            raise ValueError(f"Unknown fault type: {fault_type}")
            
        fault_freq = fault_freq_map[fault_type]
        
        # Generate fault signature (impulsive nature)
        fault_amplitude = fault_severity * 0.8
        
        # Create periodic impulses at fault frequency
        impulse_times = np.arange(0, duration, 1/fault_freq)
        fault_signal = np.zeros_like(t)
        
        for impulse_time in impulse_times:
            if impulse_time < duration:
                # Find closest time index
                idx = np.argmin(np.abs(t - impulse_time))
                
                # Create exponentially decaying impulse
                impulse_duration = 0.001  # 1ms impulse
                impulse_indices = np.where((t >= impulse_time) & (t <= impulse_time + impulse_duration))[0]
                
                if len(impulse_indices) > 0:
                    decay_times = t[impulse_indices] - impulse_time
                    impulse = fault_amplitude * np.exp(-decay_times / (impulse_duration/5))
                    fault_signal[impulse_indices] += impulse
        
        # Add harmonics of fault frequency
        for harmonic in [2, 3, 4]:
            if harmonic * fault_freq < self.fs / 2:  # Avoid aliasing
                harmonic_amp = fault_amplitude / harmonic * 0.3
                fault_signal += harmonic_amp * np.sin(2 * np.pi * harmonic * fault_freq * t)
        
        # Combine healthy and fault signals
        total_signal = healthy_signal + fault_signal
        
        # Final verification
        assert len(t) == len(total_signal), "Time and signal arrays have different lengths"
        assert np.all(np.isfinite(total_signal)), "Signal contains non-finite values"
        
        return t, total_signal
    
    def save_to_csv(self, time_data: np.ndarray, signal_data: np.ndarray, 
                   filename: str, metadata: Dict = None):
        """
        Save signal data to CSV file with verification
        
        Args:
            time_data: Time array
            signal_data: Signal amplitude array
            filename: Output filename
            metadata: Optional metadata dictionary
        """
        # Verify input data quality
        assert len(time_data) == len(signal_data), "Time and signal arrays must have same length"
        assert np.all(np.isfinite(time_data)), "Time data contains non-finite values"
        assert np.all(np.isfinite(signal_data)), "Signal data contains non-finite values"
        assert np.all(np.diff(time_data) > 0), "Time array is not monotonically increasing"
        
        # Create DataFrame
        df = pd.DataFrame({
            'Time_s': time_data,
            'Amplitude_g': signal_data  # Assuming acceleration in g units
        })
        
        # Add metadata as header comments if provided
        output_path = os.path.join('test_data', filename)
        os.makedirs('test_data', exist_ok=True)
        
        with open(output_path, 'w') as f:
            if metadata:
                f.write(f"# Bearing Vibration Data\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("#\n")
            
            # Write the DataFrame
            df.to_csv(f, index=False)
        
        # Verify the saved file by reading it back
        try:
            df_verify = pd.read_csv(output_path, comment='#')
            time_verify = df_verify['Time_s'].values
            signal_verify = df_verify['Amplitude_g'].values
            
            # Check if the data was saved and loaded correctly
            assert len(time_verify) == len(time_data), "Saved data length mismatch"
            assert np.allclose(time_verify, time_data, rtol=1e-10), "Time data mismatch after save/load"
            assert np.allclose(signal_verify, signal_data, rtol=1e-10), "Signal data mismatch after save/load"
            assert np.all(np.diff(time_verify) > 0), "Saved time array is not monotonic"
            
            print(f"✅ Data saved and verified: {output_path}")
            
        except Exception as e:
            print(f"❌ Error verifying saved file {output_path}: {e}")
            # Don't raise here, just warn
        
        print(f"Data saved to: {output_path}")

def generate_test_dataset():
    """Generate a complete test dataset for the vibration analyzer"""
    
    print("Generating test dataset with improved data quality checks...")
    
    generator = BearingDataGenerator()
    
    # Test conditions
    test_cases = [
        {'rpm': 1800, 'duration': 2.0, 'condition': 'healthy', 'noise': 0.1},
        {'rpm': 1800, 'duration': 2.0, 'condition': 'inner_race', 'severity': 0.3, 'noise': 0.15},
        {'rpm': 1800, 'duration': 2.0, 'condition': 'outer_race', 'severity': 0.5, 'noise': 0.12},
        {'rpm': 3600, 'duration': 1.5, 'condition': 'healthy', 'noise': 0.08},
        {'rpm': 3600, 'duration': 1.5, 'condition': 'ball', 'severity': 0.4, 'noise': 0.1},
        {'rpm': 1200, 'duration': 2.0, 'condition': 'cage', 'severity': 0.6, 'noise': 0.2}
    ]
    
    print(f"Generating {len(test_cases)} test cases...")
    
    successful_files = 0
    failed_files = 0
    
    for i, case in enumerate(test_cases):
        rpm = case['rpm']
        duration = case['duration']
        condition = case['condition']
        noise = case['noise']
        
        print(f"\nProcessing case {i+1}/{len(test_cases)}: {condition} at {rpm} RPM")
        
        try:
            # Calculate bearing frequencies for metadata
            freqs = generator.calculate_bearing_frequencies(rpm)
            
            metadata = {
                'RPM': rpm,
                'Duration_s': duration,
                'Condition': condition,
                'Sampling_Rate_Hz': generator.fs,
                'Shaft_Frequency_Hz': f"{freqs['shaft_frequency']:.2f}",
                'BPFO_Hz': f"{freqs['bpfo']:.2f}",
                'BPFI_Hz': f"{freqs['bpfi']:.2f}",
                'BSF_Hz': f"{freqs['bsf']:.2f}",
                'FTF_Hz': f"{freqs['ftf']:.2f}"
            }
            
            if condition == 'healthy':
                t, signal = generator.generate_healthy_signal(rpm, duration, noise)
                filename = f"bearing_{condition}_{rpm}rpm_{i+1:02d}.csv"
            else:
                severity = case['severity']
                t, signal = generator.generate_fault_signal(rpm, duration, condition, severity, noise)
                filename = f"bearing_{condition}_{rpm}rpm_sev{int(severity*100)}_{i+1:02d}.csv"
                metadata['Fault_Severity'] = severity
            
            # Additional verification before saving
            print(f"   Generated {len(t)} samples over {duration}s")
            print(f"   Time range: {t[0]:.6f} to {t[-1]:.6f} seconds")
            print(f"   Signal range: {np.min(signal):.4f} to {np.max(signal):.4f}")
            
            generator.save_to_csv(t, signal, filename, metadata)
            successful_files += 1
            
        except Exception as e:
            print(f"   ❌ ERROR generating {condition} case: {e}")
            failed_files += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print(f"GENERATION SUMMARY:")
    print(f"✅ Successful: {successful_files}/{len(test_cases)} files")
    print(f"❌ Failed: {failed_files}/{len(test_cases)} files")
    
    if successful_files > 0:
        print(f"\nGenerated files are in 'test_data' directory")
        print("Files contain realistic bearing signatures with known fault frequencies")
        print("\nTo verify data quality, run: python diagnose_data.py")
    
    return successful_files, failed_files

if __name__ == "__main__":
    print("=" * 60)
    print("VIBRATION TEST DATA GENERATOR")
    print("=" * 60)
    
    try:
        successful, failed = generate_test_dataset()
        
        if successful > 0:
            print(f"\n🎉 SUCCESS: Generated {successful} test files!")
            print("\nNext steps:")
            print("1. Run: python diagnose_data.py  (to verify data quality)")
            print("2. Run: python main.py  (to test the application)")
        else:
            print(f"\n❌ FAILED: No files were generated successfully")
            print("Check the error messages above for details")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your Python environment and try again")