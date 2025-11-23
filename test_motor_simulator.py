"""
Test script for motor analysis simulator
"""

import sys
import numpy as np

# Test imports
try:
    import tkinter as tk
    from tkinter import ttk
    print("✓ Tkinter imported successfully")
except ImportError as e:
    print(f"✗ Tkinter import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")
    sys.exit(1)

# Test motor analysis classes
try:
    from motor_analysis_simulator import MotorEconomicAnalysis, MotorDynamicSimulator
    print("✓ Motor classes imported successfully")
except ImportError as e:
    print(f"✗ Motor classes import failed: {e}")
    sys.exit(1)

# Test economic analysis
print("\n" + "="*60)
print("TESTING ECONOMIC ANALYSIS")
print("="*60)

try:
    analysis = MotorEconomicAnalysis()
    results = analysis.analyze_motors()

    print(f"\nMotor A Present Worth: Rs. {results['Motor A']['present_worth']:,.2f}")
    print(f"Motor B Present Worth: Rs. {results['Motor B']['present_worth']:,.2f}")
    print(f"\nRecommendation: {results['recommendation']}")
    print(f"Savings: Rs. {results['savings']:,.2f}")

    print("\n✓ Economic analysis completed successfully")
except Exception as e:
    print(f"✗ Economic analysis failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dynamic simulator
print("\n" + "="*60)
print("TESTING DYNAMIC SIMULATOR")
print("="*60)

try:
    simulator = MotorDynamicSimulator()

    # Run a few simulation steps
    print("\nRunning simulation steps...")
    for i in range(10):
        speed, torque, current, power = simulator.simulate_step(method='rk45')

    print(f"Time: {simulator.time:.4f} s")
    print(f"Speed: {speed:.2f} RPM")
    print(f"Torque: {torque:.2f} N.m")
    print(f"Current: {current:.2f} A")
    print(f"Power: {power/1000:.2f} kW")

    print("\n✓ Dynamic simulator completed successfully")
except Exception as e:
    print(f"✗ Dynamic simulator failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Euler method
print("\n" + "="*60)
print("TESTING EULER METHOD")
print("="*60)

try:
    simulator2 = MotorDynamicSimulator()

    for i in range(10):
        speed, torque, current, power = simulator2.simulate_step(method='euler')

    print(f"Time: {simulator2.time:.4f} s")
    print(f"Speed: {speed:.2f} RPM")
    print(f"Torque: {torque:.2f} N.m")

    print("\n✓ Euler method completed successfully")
except Exception as e:
    print(f"✗ Euler method failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("="*60)
print("\nTo run the GUI application, execute:")
print("    python3 motor_analysis_simulator.py")
