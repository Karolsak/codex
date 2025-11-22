# Advanced Electrical Engineering Simulator

A comprehensive Python application for electrical engineering calculations and simulations using Tkinter GUI.

## Features

### 1. Tariff Calculator
Calculates electricity tariff and related parameters for power supply undertakings:
- **Input Parameters:**
  - Energy generated (kWh/year)
  - Maximum demand (MW)
  - Cost breakdown (Fuel, Generation, Transmission, Distribution)
  - Running charges percentages
  - System losses percentage

- **Calculated Results:**
  - Total costs (Running charges and Fixed charges breakdown)
  - Energy analysis (Generated, Losses, Available to consumers)
  - Load factor
  - Tariff structure (Fixed charge per kW/year, Running charge per kWh)
  - Overall cost per unit

### 2. Generator Dynamics Simulator
Real-time simulation of power system generator dynamics using swing equation:
- **Differential Equations:** Models generator rotor dynamics
- **Parameters:**
  - Inertia constant H (MJ/MVA)
  - Damping coefficient D
  - Mechanical power Pm (per unit)
  - Initial rotor angle
- **ODE Solvers:** RK45 (Runge-Kutta 4th/5th order) and Euler method
- **Visualization:** Real-time plots of rotor angle and speed deviation

### 3. RLC Circuit Dynamics
Simulates series RLC circuit behavior with AC source:
- **Parameters:**
  - Resistance R (Ω)
  - Inductance L (H)
  - Capacitance C (F)
  - Source voltage V (V)
- **ODE Solvers:** RK45 and Euler method
- **Visualization:** Current and capacitor voltage over time

## Problem Solved

The application solves the following tariff calculation problem:

**Given:**
- Energy generated: 39 × 10^7 kWh/year
- Maximum demand: 130 MW
- Costs: Fuel (Rs. 37.5 lakhs), Generation (Rs. 18 lakhs), Transmission (Rs. 37.5 lakhs), Distribution (Rs. 25.5 lakhs)
- Running charges: 90%, 10%, 5%, 7% respectively
- System losses: 10%

**Results:**
- Load Factor: **30.82%**
- Overall Cost per Unit: **0.0338 Rs/kWh** or **3.38 paise/kWh**
- Fixed Charge: **61.00 Rs/kW/year**
- Running Charge: **0.0117 Rs/kWh**

## Requirements

```bash
pip install numpy matplotlib
```

Python standard library: tkinter (usually included with Python)

## Usage

Run the application:

```bash
python3 electrical_engineering_simulator.py
```

### Interface Guide

**Tariff Calculator Tab:**
1. Enter input parameters or use default values
2. Click "Calculate Tariff" button
3. View detailed results in the right panel

**Generator Dynamics Tab:**
1. Adjust parameters using sliders:
   - Inertia constant H
   - Damping coefficient D
   - Mechanical power Pm
   - Initial rotor angle
2. Select ODE solver (RK45 or Euler)
3. Click "Start" to run simulation
4. Click "Stop" to halt simulation
5. Click "Reset" to clear plots

**RLC Circuit Dynamics Tab:**
1. Adjust circuit parameters using sliders:
   - Resistance R
   - Inductance L
   - Capacitance C
   - Source voltage V
2. Select ODE solver
3. Use Start/Stop/Reset buttons for simulation control

## Key Features

### Advanced GUI Components
- **Multi-tab interface** for organized functionality
- **Interactive sliders** for real-time parameter adjustment
- **Professional layout** with labeled frames
- **Responsive design** with auto-scaling support

### Mathematical Modeling
- **Swing Equation:** Models synchronous generator dynamics
  ```
  d²δ/dt² = (ωs/2H) × (Pm - Pe - D×dδ/dt)
  ```
- **RLC Circuit Equations:**
  ```
  di/dt = (Vs - Vc - R×i) / L
  dVc/dt = i / C
  ```

### ODE Solvers
1. **RK45 (Runge-Kutta 4th/5th order):**
   - High accuracy
   - Better for stiff equations
   - Recommended for production use

2. **Euler Method:**
   - Simple first-order method
   - Fast computation
   - Good for educational purposes

### Visualization
- **Real-time plotting** using matplotlib
- **Auto-scaling** for different window sizes
- **Multiple subplots** for comprehensive analysis
- **Grid lines** for better readability

## Practical Applications in Electrical Engineering

1. **Power System Planning:** Calculate tariffs for new power plants
2. **Stability Analysis:** Study generator response to disturbances
3. **Circuit Design:** Analyze RLC circuit transient behavior
4. **Education:** Demonstrate power system concepts
5. **Load Management:** Optimize load factor for cost reduction

## Technical Details

### Tariff Calculation Formula
```
Annual Bill = Fixed Charge × Max Demand (kW) + Running Charge × Energy Consumed (kWh)
```

### Load Factor
```
Load Factor = (Average Demand / Maximum Demand) × 100%
```

### Energy Balance
```
Energy to Consumers = Energy Generated × (1 - Loss Percentage/100)
```

## Window Management
- Minimum window size: 1000 × 700 pixels
- Default size: 1400 × 900 pixels
- Auto-scaling when window is resized
- Responsive layout using grid and pack geometry managers

## Author Notes
This application combines theoretical electrical engineering concepts with practical computational tools, making it suitable for:
- Power system engineers
- Electrical engineering students
- Researchers in power systems
- Utility companies for tariff planning

## Future Enhancements
- Additional power system models (load flow, fault analysis)
- Export results to CSV/PDF
- Save/Load simulation configurations
- Three-phase system analysis
- Power quality analysis tools
