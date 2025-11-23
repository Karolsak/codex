# Advanced Motor Replacement Analysis & Dynamic Simulation System

A comprehensive Python application combining economic analysis with real-time motor dynamics simulation for electrical engineering applications.

## 📋 Overview

This application solves the motor replacement problem and provides an advanced Tkinter-based GUI for:
- **Economic Analysis**: Compare motors based on present worth analysis
- **Dynamic Simulation**: Real-time motor behavior using differential equations
- **ODE Solvers**: Both RK45 (Runge-Kutta) and Euler methods
- **Visualization**: Interactive plots with matplotlib
- **Control Panel**: Adjustable parameters with sliders
- **Responsive Design**: Auto-scaling GUI components

## 🎯 Problem Statement

A 22.5 kW condensate pump motor has burnt beyond economical repairs. Two replacement alternatives:

**Motor A**
- Cost: Rs. 6,000
- Efficiency @ full-load: 90%
- Efficiency @ half-load: 86%
- Annual maintenance: Rs. 420

**Motor B**
- Cost: Rs. 4,000
- Efficiency @ full-load: 85%
- Efficiency @ half-load: 82%
- Annual maintenance: Rs. 240

**Operating Conditions**
- Life: 20 years
- Salvage value: 10% of initial cost
- Interest rate: 5% annually
- Full-load operation: 25% of time
- Half-load operation: 75% of time
- Energy rate: 10 paise per kWh

## 📊 Solution

**RECOMMENDATION: Motor A**
- **Total Present Worth (Motor A)**: Rs. 186,344.77
- **Total Present Worth (Motor B)**: Rs. 191,415.21
- **Savings with Motor A**: Rs. 5,070.44 over 20 years

**Justification**: Although Motor A has a higher initial cost (Rs. 6,000 vs Rs. 4,000), its superior efficiency (90%/86% vs 85%/82%) results in significantly lower operating costs over the 20-year lifetime, leading to a lower total present worth.

## 🚀 Features

### 1. Economic Analysis Module
- Present worth calculations
- Life cycle cost analysis
- Energy consumption analysis
- Comprehensive reporting
- Customizable parameters

### 2. Dynamic Simulation Module
- Induction motor mathematical model
- ODE solvers (RK45 and Euler)
- Real-time state variables:
  - Motor speed (RPM)
  - Electromagnetic torque (N.m)
  - Stator current (A)
  - Output power (kW)
- Adjustable load conditions
- Interactive parameter tuning

### 3. GUI Features
- **Multi-tab Interface**:
  - Tab 1: Economic Analysis
  - Tab 2: Dynamic Simulation
  - Tab 3: Comprehensive Results

- **Control Panel**:
  - Load torque slider (0-200 N.m)
  - Voltage slider (0-500 V)
  - Inertia slider (0.1-2.0 kg.m²)
  - ODE solver selection (RK45/Euler)

- **Visualization**:
  - Real-time plotting
  - Speed vs time
  - Torque vs time
  - Current vs time
  - Power vs time
  - Cost comparison charts
  - Energy consumption graphs
  - Cumulative cost analysis
  - Cost breakdown pie charts

- **Buttons**:
  - Start simulation
  - Stop simulation
  - Reset simulation
  - Calculate analysis
  - Generate comparisons

- **Responsive Design**:
  - Auto-scaling plots
  - Window resize handling
  - Dynamic layout adjustment

## 📦 Installation

### Prerequisites

```bash
# Python 3.7 or higher
python3 --version

# Install required packages
pip3 install numpy matplotlib

# For GUI (Linux)
sudo apt-get install python3-tk

# For GUI (macOS)
brew install python-tk

# For GUI (Windows)
# Tkinter is usually included with Python installation
```

### Quick Start

```bash
# Clone or download the files
cd motor_analysis

# Install dependencies
pip3 install -r requirements.txt

# Run the GUI application
python3 motor_analysis_simulator.py

# Or run core analysis only (no GUI)
python3 motor_analysis_core.py
```

## 💻 Usage

### GUI Application

```python
# Run the full GUI application
python3 motor_analysis_simulator.py
```

**Economic Analysis Tab:**
1. Enter motor parameters (or use defaults)
2. Click "Calculate" to run analysis
3. View detailed results in the text area
4. Click "Reset" to restore defaults

**Dynamic Simulation Tab:**
1. Select ODE solver (RK45 or Euler)
2. Adjust parameters using sliders
3. Click "Start" to begin simulation
4. Watch real-time plots update
5. Click "Stop" to pause
6. Click "Reset" to clear and restart

**Comprehensive Results Tab:**
1. Switch to this tab
2. Click "Generate Comprehensive Analysis"
3. View comparative charts and graphs

### Core Module (Command Line)

```python
# Run core analysis without GUI
python3 motor_analysis_core.py

# Or use in your own scripts
from motor_analysis_core import MotorEconomicAnalysis, MotorDynamicSimulator

# Economic analysis
analysis = MotorEconomicAnalysis()
results = analysis.analyze_motors()
print(f"Recommended: {results['recommendation']}")
print(f"Savings: Rs. {results['savings']:,.2f}")

# Dynamic simulation
simulator = MotorDynamicSimulator()
for i in range(1000):
    speed, torque, current, power = simulator.simulate_step(method='rk45')
print(f"Final speed: {speed:.2f} RPM")
```

## 🔬 Technical Details

### Economic Analysis

**Formulas Used:**

1. **Annual Energy Cost**:
   ```
   E_full = (P / η_full) × hours_full
   E_half = (P × 0.5 / η_half) × hours_half
   Annual_cost = (E_full + E_half) × rate
   ```

2. **Present Worth Factor**:
   ```
   PW_factor = ((1 + i)^n - 1) / (i × (1 + i)^n)
   ```

3. **Total Present Worth**:
   ```
   PW_total = Initial_cost + (Annual_cost × PW_factor) - PW_salvage
   ```

### Dynamic Simulation

**Motor Model**: Simplified induction motor in d-q reference frame

**State Variables**:
- `ids, iqs`: Stator currents (d-q axes)
- `idr, iqr`: Rotor currents (d-q axes)
- `omega`: Angular velocity
- `theta`: Rotor position

**Differential Equations**:
```
dids/dt = (vds - Rs×ids + ωs×Ls×iqs) / Lσs
diqs/dt = (vqs - Rs×iqs - ωs×Ls×ids) / Lσs
didr/dt = (-Rr×idr + ωslip×Lr×iqr) / Lσr
diqr/dt = (-Rr×iqr - ωslip×Lr×idr) / Lσr
dω/dt = (Te - Tload - B×ω) / J
```

**Electromagnetic Torque**:
```
Te = 1.5 × P × Lm × (iqs×idr - ids×iqr)
```

### ODE Solvers

**RK45 (Runge-Kutta 4th Order)**:
```python
k1 = f(t, y)
k2 = f(t + dt/2, y + dt×k1/2)
k3 = f(t + dt/2, y + dt×k2/2)
k4 = f(t + dt, y + dt×k3)
y_new = y + (dt/6) × (k1 + 2×k2 + 2×k3 + k4)
```

**Euler Method**:
```python
y_new = y + dt × f(t, y)
```

## 📁 File Structure

```
motor_analysis/
├── motor_analysis_simulator.py    # Full GUI application
├── motor_analysis_core.py          # Core modules (no GUI)
├── requirements.txt                # Python dependencies
├── README_MOTOR_SIMULATOR.md       # This file
└── test_core_functionality.py      # Test script
```

## 🎨 GUI Screenshots Description

### Economic Analysis Tab
- Left panel: Input fields for all motor parameters
- Right panel: Detailed analysis results with recommendations
- Buttons: Calculate, Reset

### Dynamic Simulation Tab
- Top: Control panel with solver selection and action buttons
- Left: Parameter sliders (Load, Voltage, Inertia) and real-time data display
- Right: Four real-time plots (Speed, Torque, Current, Power)

### Comprehensive Results Tab
- Four comparison charts:
  - Cost comparison bar chart
  - Energy consumption comparison
  - Cumulative cost over lifetime
  - Cost breakdown pie chart

## 🔧 Customization

### Modify Motor Parameters

```python
# In motor_analysis_core.py or motor_analysis_simulator.py

# Economic parameters
analysis = MotorEconomicAnalysis()
analysis.motor_power = 30.0  # Change power to 30 kW
analysis.life_years = 15     # Change life to 15 years
analysis.interest_rate = 0.07  # Change interest to 7%

# Dynamic simulation parameters
simulator = MotorDynamicSimulator()
simulator.rated_power = 30e3   # 30 kW
simulator.load_torque = 100    # 100 N.m
simulator.J = 1.0              # 1.0 kg.m²
```

### Add New Features

The modular design allows easy extension:
- Add new motor types
- Implement different cost models
- Add more visualization options
- Extend simulation models

## 📊 Output Examples

### Economic Analysis Output
```
================================================================================
MOTOR REPLACEMENT ECONOMIC ANALYSIS REPORT
================================================================================
Analysis Date: 2025-11-23 07:50:58
Analysis Period: 20 years
Interest Rate: 5.00%

MOTOR A - DETAILED ANALYSIS
Initial Cost:              Rs. 6,000.00
Annual Energy Cost:        Rs. 14,069.48
Total Present Worth:       Rs. 186,344.77

MOTOR B - DETAILED ANALYSIS
Initial Cost:              Rs. 4,000.00
Annual Energy Cost:        Rs. 14,810.78
Total Present Worth:       Rs. 191,415.21

RECOMMENDATION: Motor A
Savings: Rs. 5,070.44
================================================================================
```

### Dynamic Simulation Output
```
Time: 0.500 s
Solver: RK45
----------------------------------------
Speed:   -475.09 RPM
Torque:  0.00 N.m
Current: 18.98 A
Power:   -0.00 kW
----------------------------------------
```

## 🎓 Educational Value

This application is valuable for:
- **Electrical Engineering Students**: Understanding motor economics and dynamics
- **Power Systems Engineers**: Motor replacement analysis
- **Control Systems**: Dynamic simulation and ODE solving
- **Python Programming**: GUI development with Tkinter and matplotlib
- **Numerical Methods**: Comparing RK45 vs Euler solvers

## 🐛 Troubleshooting

### Issue: Tkinter not found
```bash
# Linux
sudo apt-get install python3-tk

# macOS
brew install python-tk
```

### Issue: Matplotlib backend error
```python
# Add to top of script
import matplotlib
matplotlib.use('TkAgg')
```

### Issue: NumPy not found
```bash
pip3 install --upgrade numpy
```

## 📝 License

This project is provided for educational and professional use. Feel free to modify and extend.

## 👨‍💻 Author

Created as an advanced electrical engineering simulation tool combining economic analysis with dynamic motor modeling.

## 🔍 Keywords

Motor replacement, economic analysis, present worth, induction motor, dynamic simulation, ODE solver, RK45, Euler method, Tkinter GUI, matplotlib, electrical engineering, power systems, Python simulation, real-time visualization

## 📚 References

- Induction motor theory and modeling
- Present worth analysis in engineering economics
- Numerical methods for differential equations
- Python GUI development with Tkinter
- Matplotlib for scientific visualization

---

**For questions, issues, or suggestions, please refer to the code comments or extend the application as needed.**
