"""
Test core functionality without GUI components
"""

import sys
import numpy as np

print("Testing Motor Analysis Core Functionality")
print("=" * 70)

# Test economic analysis
print("\n1. Testing Economic Analysis Module...")
print("-" * 70)

try:
    # Import only the economic analysis class
    exec(open('motor_analysis_simulator.py').read().split('class MotorAnalysisGUI')[0])

    analysis = MotorEconomicAnalysis()
    results = analysis.analyze_motors()

    print("\nMOTOR REPLACEMENT ANALYSIS RESULTS")
    print("=" * 70)

    print("\nMOTOR A:")
    print(f"  Initial Cost: Rs. {results['Motor A']['initial_cost']:,.2f}")
    print(f"  Annual Energy Cost: Rs. {results['Motor A']['annual_energy_cost']:,.2f}")
    print(f"  Annual Energy Consumption: {results['Motor A']['annual_energy_kwh']:,.2f} kWh")
    print(f"  Annual Maintenance: Rs. {results['Motor A']['annual_maintenance']:,.2f}")
    print(f"  Total Annual Cost: Rs. {results['Motor A']['total_annual_cost']:,.2f}")
    print(f"  Salvage Value: Rs. {results['Motor A']['salvage_value']:,.2f}")
    print(f"  Present Worth: Rs. {results['Motor A']['present_worth']:,.2f}")

    print("\nMOTOR B:")
    print(f"  Initial Cost: Rs. {results['Motor B']['initial_cost']:,.2f}")
    print(f"  Annual Energy Cost: Rs. {results['Motor B']['annual_energy_cost']:,.2f}")
    print(f"  Annual Energy Consumption: {results['Motor B']['annual_energy_kwh']:,.2f} kWh")
    print(f"  Annual Maintenance: Rs. {results['Motor B']['annual_maintenance']:,.2f}")
    print(f"  Total Annual Cost: Rs. {results['Motor B']['total_annual_cost']:,.2f}")
    print(f"  Salvage Value: Rs. {results['Motor B']['salvage_value']:,.2f}")
    print(f"  Present Worth: Rs. {results['Motor B']['present_worth']:,.2f}")

    print("\n" + "=" * 70)
    print(f"RECOMMENDATION: {results['recommendation']}")
    print(f"SAVINGS OVER 20 YEARS: Rs. {results['savings']:,.2f}")
    print("=" * 70)

    if results['recommendation'] == 'Motor A':
        print("\nJustification: Motor A has higher efficiency which results in")
        print("lower operating costs over the lifetime, despite higher initial cost.")
    else:
        print("\nJustification: Motor B has lower total cost over the lifetime")
        print("when considering initial cost, operating costs, and maintenance.")

    print("\n✓ Economic analysis test PASSED")

except Exception as e:
    print(f"✗ Economic analysis test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dynamic simulator
print("\n\n2. Testing Dynamic Simulator Module...")
print("-" * 70)

try:
    simulator = MotorDynamicSimulator()

    print("\nSimulator initialized with parameters:")
    print(f"  Rated Power: {simulator.rated_power/1000:.1f} kW")
    print(f"  Rated Voltage: {simulator.rated_voltage} V")
    print(f"  Rated Speed: {simulator.rated_speed} RPM")
    print(f"  Moment of Inertia: {simulator.J} kg.m²")
    print(f"  Load Torque: {simulator.load_torque} N.m")

    print("\nRunning simulation with RK45 solver...")
    for i in range(100):
        speed, torque, current, power = simulator.simulate_step(method='rk45')

    print(f"\nAfter {simulator.time:.3f} seconds:")
    print(f"  Speed: {speed:.2f} RPM")
    print(f"  Electromagnetic Torque: {torque:.2f} N.m")
    print(f"  Stator Current: {current:.2f} A")
    print(f"  Output Power: {power/1000:.2f} kW")
    print(f"  Data points collected: {len(simulator.time_data)}")

    # Test Euler method
    print("\nResetting and running with Euler method...")
    simulator.reset_state()

    for i in range(100):
        speed, torque, current, power = simulator.simulate_step(method='euler')

    print(f"\nAfter {simulator.time:.3f} seconds (Euler):")
    print(f"  Speed: {speed:.2f} RPM")
    print(f"  Electromagnetic Torque: {torque:.2f} N.m")

    print("\n✓ Dynamic simulator test PASSED")

except Exception as e:
    print(f"✗ Dynamic simulator test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Performance comparison
print("\n\n3. Testing Different Load Conditions...")
print("-" * 70)

try:
    # Test with different loads
    loads = [25, 50, 75, 100]
    print("\nSpeed response for different load torques (at t=0.1s):")
    print(f"{'Load (N.m)':<12} {'Speed (RPM)':<15} {'Current (A)':<15}")
    print("-" * 42)

    for load in loads:
        sim = MotorDynamicSimulator()
        sim.load_torque = load
        for i in range(100):
            speed, torque, current, power = sim.simulate_step(method='rk45')
        print(f"{load:<12} {speed:<15.2f} {current:<15.2f}")

    print("\n✓ Load condition test PASSED")

except Exception as e:
    print(f"✗ Load condition test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL CORE FUNCTIONALITY TESTS PASSED!")
print("=" * 70)
print("\nNote: GUI tests skipped (requires display)")
print("To run the full GUI application:")
print("  python3 motor_analysis_simulator.py")
print("\nMake sure you have:")
print("  - Tkinter installed (python3-tk)")
print("  - A display environment (X11, Wayland, etc.)")
