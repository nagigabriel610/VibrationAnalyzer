"""
Diagnostic and Fix for Linear vs Log Scale Issue
Let's find and fix the exact problem
"""

import os

def find_plotting_method():
    """Find where the plotting method is in the file"""
    
    file_path = 'src/gui/main_window_enhanced.py'
    
    if not os.path.exists(file_path):
        print("❌ File not found: src/gui/main_window_enhanced.py")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the plotting method
    plot_method_line = None
    scale_logic_lines = []
    
    for i, line in enumerate(lines, 1):
        if 'def plot_frequency_domain_interactive' in line:
            plot_method_line = i
            print(f"✅ Found plotting method at line {i}")
        
        if 'amplitude_scale ==' in line and 'Logarithmic' in line:
            scale_logic_lines.append(i)
            print(f"🔍 Found scale logic at line {i}: {line.strip()}")
    
    # Show context around scale logic
    if scale_logic_lines:
        print("\n📋 CURRENT SCALE LOGIC:")
        for line_num in scale_logic_lines:
            start = max(0, line_num - 3)
            end = min(len(lines), line_num + 5)
            
            print(f"\nAround line {line_num}:")
            for i in range(start, end):
                marker = ">>> " if i == line_num - 1 else "    "
                print(f"{marker}{i+1:3d}: {lines[i].rstrip()}")
    
    return plot_method_line, scale_logic_lines

def create_debug_version():
    """Create a version with debug prints to see what's happening"""
    
    file_path = 'src/gui/main_window_enhanced.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the plotting logic with debug version
    old_plotting_logic = '''        # Plot based on amplitude scale
        if amplitude_scale == "Logarithmic (dB)":
            self.ax.plot(frequencies, magnitude, 'b-', linewidth=1, label='Magnitude (dB)')
            self.ax.set_ylabel('Magnitude (dB)')
        else:
            self.ax.semilogy(frequencies, magnitude, 'b-', linewidth=1, label='Magnitude')
            self.ax.set_ylabel('Magnitude')'''
    
    new_plotting_logic = '''        # Plot based on amplitude scale - DEBUG VERSION
        print(f"🔍 DEBUG: amplitude_scale = '{amplitude_scale}'")
        print(f"🔍 DEBUG: magnitude range = {np.min(magnitude):.6f} to {np.max(magnitude):.6f}")
        
        if amplitude_scale == "Logarithmic (dB)":
            print("🔍 DEBUG: Using dB mode - linear axis with dB values")
            self.ax.plot(frequencies, magnitude, 'b-', linewidth=1, label='Magnitude (dB)')
            self.ax.set_ylabel('Magnitude (dB)')
            self.ax.set_yscale('linear')
            print(f"🔍 DEBUG: Y-axis set to linear, ylabel = 'Magnitude (dB)'")
        else:
            print("🔍 DEBUG: Using linear mode - linear axis with linear values")
            self.ax.plot(frequencies, magnitude, 'b-', linewidth=1, label='Magnitude')
            self.ax.set_ylabel('Magnitude')
            self.ax.set_yscale('linear')
            print(f"🔍 DEBUG: Y-axis set to linear, ylabel = 'Magnitude'")'''
    
    # Try to replace
    if old_plotting_logic.replace(' ', '').replace('\n', '') in content.replace(' ', '').replace('\n', ''):
        new_content = content.replace(old_plotting_logic, new_plotting_logic)
        
        # Write debug version
        with open('src/gui/main_window_debug.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Created debug version: src/gui/main_window_debug.py")
        return True
    else:
        print("❌ Could not find exact plotting logic to replace")
        print("Let's look at what exists...")
        
        # Search for any amplitude_scale logic
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'amplitude_scale' in line and ('Logarithmic' in line or '==' in line):
                print(f"Line {i+1}: {line.strip()}")
        
        return False

def create_simple_fix():
    """Create a simple, guaranteed fix"""
    
    simple_fix = '''
# SIMPLE FIX - Add this to plot_frequency_domain_interactive method
# Replace the entire plotting section with this:

        # SIMPLE LINEAR VS LOG FIX
        if amplitude_scale == "Logarithmic (dB)":
            # Convert linear magnitude to dB if not already done
            if np.max(magnitude) > 10:  # If values are large, they're linear
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
                self.ax.plot(frequencies, magnitude_db, 'b-', linewidth=1, label='Magnitude (dB)')
                self.ax.set_ylabel('Magnitude (dB)')
                print(f"🔍 Converted to dB: range {np.min(magnitude_db):.1f} to {np.max(magnitude_db):.1f} dB")
            else:
                # Already in dB
                self.ax.plot(frequencies, magnitude, 'b-', linewidth=1, label='Magnitude (dB)')
                self.ax.set_ylabel('Magnitude (dB)')
                print(f"🔍 Already dB: range {np.min(magnitude):.1f} to {np.max(magnitude):.1f} dB")
            self.ax.set_yscale('linear')
        else:
            # Linear scale
            self.ax.plot(frequencies, magnitude, 'b-', linewidth=1, label='Magnitude')
            self.ax.set_ylabel('Magnitude')
            self.ax.set_yscale('linear')
            print(f"🔍 Linear mode: range {np.min(magnitude):.6f} to {np.max(magnitude):.6f}")
    '''
    
    with open('simple_scale_fix.txt', 'w') as f:
        f.write(simple_fix)
    
    print("✅ Created simple_scale_fix.txt with manual fix instructions")

def main():
    """Main diagnostic function"""
    print("🔍 DIAGNOSING LINEAR VS LOG SCALE ISSUE")
    print("=" * 50)
    
    # Step 1: Find where the plotting logic is
    plot_line, scale_lines = find_plotting_method()
    
    print("\n" + "=" * 50)
    
    # Step 2: Try to create debug version
    if create_debug_version():
        print("\n✅ DEBUG VERSION CREATED")
        print("\n📋 NEXT STEPS:")
        print("1. Test the debug version:")
        print('   python -c "from src.gui.main_window_debug import main; main()"')
        print("\n2. Look at the console output when you change between Linear and dB modes")
        print("   The debug prints will show what's happening")
        
    else:
        print("\n❌ Could not create automatic debug version")
        create_simple_fix()
        print("\n📋 MANUAL FIX REQUIRED:")
        print("1. Open: src/gui/main_window_enhanced.py")
        print("2. Find: plot_frequency_domain_interactive method")
        print("3. Look at: simple_scale_fix.txt for the replacement code")
    
    print("\n🎯 WHAT WE'RE LOOKING FOR:")
    print("• Linear mode should show small decimal numbers (0.001, 0.002...)")
    print("• dB mode should show negative numbers (-60, -40, -20, 0...)")
    print("• If both show the same values, the conversion isn't working")

if __name__ == "__main__":
    main()