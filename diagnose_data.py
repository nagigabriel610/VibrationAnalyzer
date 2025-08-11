"""
Data Diagnostic Tool - Check for flipped or incorrect data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def diagnose_csv_file(filepath):
    """Diagnose issues with CSV data file"""
    
    print(f"Diagnosing file: {os.path.basename(filepath)}")
    print("=" * 50)
    
    # Read raw CSV
    try:
        df = pd.read_csv(filepath, comment='#')
        print(f"✅ File loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Auto-detect columns (improved logic)
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
    
    print(f"\nColumn Detection:")
    print(f"   Time column: {time_col}")
    print(f"   Amplitude column: {amp_col}")
    
    if time_col is None or amp_col is None:
        print(f"❌ Column detection failed!")
        return
    
    # Extract data
    time_data = df[time_col].values
    signal_data = df[amp_col].values
    
    # Check time array properties
    print(f"\nTime Array Analysis:")
    print(f"   Length: {len(time_data)}")
    print(f"   First value: {time_data[0]:.6f}")
    print(f"   Last value: {time_data[-1]:.6f}")
    print(f"   Min value: {np.min(time_data):.6f}")
    print(f"   Max value: {np.max(time_data):.6f}")
    
    # Check if time is monotonically increasing
    time_diff = np.diff(time_data)
    is_increasing = np.all(time_diff > 0)
    is_decreasing = np.all(time_diff < 0)
    
    print(f"   Monotonically increasing: {is_increasing}")
    print(f"   Monotonically decreasing: {is_decreasing}")
    
    if is_decreasing:
        print(f"   🚨 PROBLEM FOUND: Time array is decreasing!")
    elif not is_increasing:
        print(f"   ⚠️  WARNING: Time array is not monotonic")
    
    # Calculate sampling properties
    if is_increasing or is_decreasing:
        dt_values = np.abs(time_diff)
        dt_mean = np.mean(dt_values)
        dt_std = np.std(dt_values)
        fs_estimated = 1.0 / dt_mean
        
        print(f"   Average dt: {dt_mean:.8f} seconds")
        print(f"   Std of dt: {dt_std:.8f} seconds")
        print(f"   Estimated sampling rate: {fs_estimated:.1f} Hz")
    
    # Check signal properties
    print(f"\nSignal Array Analysis:")
    print(f"   Length: {len(signal_data)}")
    print(f"   Min: {np.min(signal_data):.6f}")
    print(f"   Max: {np.max(signal_data):.6f}")
    print(f"   Mean: {np.mean(signal_data):.6f}")
    print(f"   RMS: {np.sqrt(np.mean(signal_data**2)):.6f}")
    print(f"   Std: {np.std(signal_data):.6f}")
    
    # Check for NaN or infinite values
    nan_time = np.sum(np.isnan(time_data))
    nan_signal = np.sum(np.isnan(signal_data))
    inf_time = np.sum(np.isinf(time_data))
    inf_signal = np.sum(np.isinf(signal_data))
    
    if nan_time + nan_signal + inf_time + inf_signal > 0:
        print(f"   ⚠️  Data quality issues:")
        print(f"      NaN in time: {nan_time}")
        print(f"      NaN in signal: {nan_signal}")
        print(f"      Inf in time: {inf_time}")
        print(f"      Inf in signal: {inf_signal}")
    else:
        print(f"   ✅ No NaN or infinite values found")
    
    # Plot the data
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain plot
    axes[0,0].plot(time_data, signal_data, 'b-', linewidth=0.5)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].set_title('Time Domain Signal')
    axes[0,0].grid(True, alpha=0.3)
    
    # Time array itself
    axes[0,1].plot(time_data, 'r.-', markersize=2)
    axes[0,1].set_xlabel('Sample Index')
    axes[0,1].set_ylabel('Time (s)')
    axes[0,1].set_title('Time Array vs Sample Index')
    axes[0,1].grid(True, alpha=0.3)
    
    # Signal histogram
    axes[1,0].hist(signal_data, bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Amplitude')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Signal Histogram')
    axes[1,0].grid(True, alpha=0.3)
    
    # Quick FFT to check frequency content
    if is_increasing:
        # Correct time array - can do FFT
        signal_ac = signal_data - np.mean(signal_data)
        fft_result = np.fft.fft(signal_ac)
        n_samples = len(signal_ac)
        freqs = np.fft.fftfreq(n_samples, dt_mean)[:n_samples//2]
        magnitude = np.abs(fft_result)[:n_samples//2] * 2 / n_samples
        
        axes[1,1].semilogy(freqs, magnitude, 'g-', linewidth=1)
        axes[1,1].set_xlabel('Frequency (Hz)')
        axes[1,1].set_ylabel('Magnitude')
        axes[1,1].set_title('FFT (Quick Check)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlim(0, min(1000, np.max(freqs)))
        
    elif is_decreasing:
        # Flipped time - show what happens
        axes[1,1].text(0.5, 0.5, 'TIME ARRAY IS FLIPPED!\nFFT will be incorrect', 
                      transform=axes[1,1].transAxes, ha='center', va='center',
                      fontsize=12, color='red', weight='bold')
        axes[1,1].set_title('FFT Status: CORRUPTED')
    else:
        axes[1,1].text(0.5, 0.5, 'TIME ARRAY NOT MONOTONIC!\nFFT cannot be computed', 
                      transform=axes[1,1].transAxes, ha='center', va='center',
                      fontsize=12, color='red', weight='bold')
        axes[1,1].set_title('FFT Status: INVALID')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'time_increasing': is_increasing,
        'time_decreasing': is_decreasing,
        'estimated_fs': fs_estimated if (is_increasing or is_decreasing) else None,
        'time_col': time_col,
        'amp_col': amp_col,
        'time_data': time_data,
        'signal_data': signal_data
    }

def fix_flipped_data(filepath, output_filepath=None):
    """Fix flipped data and save corrected version"""
    
    print(f"Fixing flipped data in: {os.path.basename(filepath)}")
    
    # Diagnose first
    diag = diagnose_csv_file(filepath)
    
    if diag['time_increasing']:
        print("✅ Time array is already correct - no fix needed")
        return
    
    if not diag['time_decreasing']:
        print("❌ Time array is not monotonic - cannot auto-fix")
        return
    
    # Fix the data
    print("🔧 Flipping time array to fix the data...")
    
    time_data = diag['time_data']
    signal_data = diag['signal_data']
    
    # Flip both arrays to maintain correct correspondence
    time_fixed = np.flip(time_data)
    signal_fixed = np.flip(signal_data)
    
    # Adjust time to start from 0
    time_fixed = time_fixed - time_fixed[0]
    
    # Create corrected DataFrame
    df_fixed = pd.DataFrame({
        'Time_s': time_fixed,
        'Amplitude_g': signal_fixed
    })
    
    # Save corrected file
    if output_filepath is None:
        base, ext = os.path.splitext(filepath)
        output_filepath = f"{base}_fixed{ext}"
    
    df_fixed.to_csv(output_filepath, index=False)
    print(f"✅ Fixed data saved to: {output_filepath}")
    
    # Verify the fix
    print("\nVerifying the fix:")
    diagnose_csv_file(output_filepath)

def check_all_test_data():
    """Check all generated test data files"""
    
    test_data_dir = "test_data"
    if not os.path.exists(test_data_dir):
        print(f"❌ Test data directory not found: {test_data_dir}")
        return
    
    csv_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No CSV files found in {test_data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to check...")
    print("=" * 60)
    
    problems_found = []
    
    for csv_file in csv_files:
        filepath = os.path.join(test_data_dir, csv_file)
        print(f"\nChecking: {csv_file}")
        
        try:
            diag = diagnose_csv_file(filepath)
            
            if diag['time_decreasing']:
                problems_found.append(csv_file)
                print(f"   🚨 PROBLEM: Time array is flipped!")
            elif not diag['time_increasing']:
                problems_found.append(csv_file)
                print(f"   ⚠️  WARNING: Time array not monotonic")
            else:
                print(f"   ✅ OK")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            problems_found.append(csv_file)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    if problems_found:
        print(f"❌ {len(problems_found)} files have problems:")
        for file in problems_found:
            print(f"   - {file}")
        print(f"\nRun fix_all_test_data() to fix automatically")
    else:
        print(f"✅ All {len(csv_files)} files are OK!")

def fix_all_test_data():
    """Fix all problematic test data files"""
    
    test_data_dir = "test_data"
    csv_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        filepath = os.path.join(test_data_dir, csv_file)
        print(f"\nProcessing: {csv_file}")
        
        try:
            # Check if it needs fixing
            df = pd.read_csv(filepath, comment='#')
            time_col = None
            for col in df.columns:
                if any(x in col.lower() for x in ['time', 't']):
                    time_col = col
                    break
            
            if time_col:
                time_data = df[time_col].values
                time_diff = np.diff(time_data)
                
                if np.all(time_diff < 0):  # Decreasing = flipped
                    print(f"   🔧 Fixing flipped data...")
                    fix_flipped_data(filepath, filepath)  # Overwrite original
                else:
                    print(f"   ✅ Already OK")
            
        except Exception as e:
            print(f"   ❌ Error processing {csv_file}: {e}")

if __name__ == "__main__":
    print("VIBRATION DATA DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Check if test data exists
    if os.path.exists("test_data"):
        print("Option 1: Check all test data files")
        print("Option 2: Check specific file")
        print("Option 3: Fix all test data files")
        
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            check_all_test_data()
        elif choice == "2":
            filepath = input("Enter CSV file path: ").strip()
            if os.path.exists(filepath):
                diagnose_csv_file(filepath)
            else:
                print(f"File not found: {filepath}")
        elif choice == "3":
            fix_all_test_data()
            print("\nRe-checking all files after fix:")
            check_all_test_data()
        else:
            print("Invalid choice")
    else:
        print("Test data directory not found.")
        print("First, let's generate clean test data...")
        
        # Try to generate test data
        try:
            print("Attempting to generate test data...")
            sys.path.append('src/data')
            from mock_generator import generate_test_dataset
            generate_test_dataset()
            print("✅ Test data generated successfully!")
            print("\nNow run this script again to check the data.")
        except Exception as e:
            print(f"❌ Error generating test data: {e}")
            print("Please run: python src/data/mock_generator.py")