# Advanced Electrical Engineering Simulator

A comprehensive Python + Tkinter GUI application for electrical engineering calculations and simulations.

## Problem Solution

### Feeder Cable Optimization Problem

**Given:**
- 500-V, 2-core feeder cable, 4 km long
- Maximum current: 200 A
- Copper loss per annum equivalent to full-load current for 6 months
- Resistance: 0.17 Ω per km per sq cm cross-section
- Cable cost: Rs. (120A + 24) per meter (A = area in sq cm)
- Interest & depreciation: 10%
- Energy cost: 4 paise per kWh

**Solution:**

The most economical cross-section is calculated by minimizing total annual cost:

**Total Annual Cost = Capital Cost Charges + Energy Loss Cost**

For 2-core cable:
- Total resistance: R = (ρ × L × 2) / A = (0.17 × 4 × 2) / A = 1.36 / A Ω
- Annual energy loss: E = I² × R × hours = (200)² × (1.36/A) × 4320 / 1000 kWh
- Capital cost: (120A + 24) × 4000 × 0.10 Rs
- Energy cost: (235008/A) × 0.04 Rs

Minimizing: dC/dA = 48000 - 9400.32/A² = 0

**Optimal Cross-Section: A = 0.4425 sq cm ≈ 0.44 sq cm**

At this cross-section:
- Total annual cost is minimized
- Perfect balance between capital and operating costs

## Features

### 1. Feeder Cable Optimizer Tab
- **Input Parameters**: Voltage, length, current, resistance, costs
- **Optimization Algorithm**: Calculates most economical cross-section
- **Visualization**: Real-time cost vs cross-section graph
- **Detailed Results**: Shows all costs and optimal design parameters

### 2. DC Motor Dynamics Tab
- **Dynamic Simulation**: Models DC motor behavior using differential equations
- **Parameters**:
  - Applied voltage
  - Armature resistance and inductance
  - Back EMF and torque constants
  - Moment of inertia and friction
- **ODE Solvers**: Choose between Euler and RK45 methods
- **Visualizations**:
  - Armature current vs time
  - Motor speed (RPM) vs time
  - Torque vs time
  - Mechanical power vs time
- **Differential Equations**:
  ```
  di/dt = (V - R·i - Ke·ω) / L
  dω/dt = (Kt·i - B·ω) / J
  ```

### 3. RLC Circuit Analysis Tab
- **AC Circuit Simulation**: Models series RLC circuit response
- **Parameters**:
  - Source voltage and frequency
  - Resistance, inductance, capacitance
- **Features**:
  - Transient response analysis
  - Resonance frequency calculation
  - Quality factor (Q) computation
  - Frequency response (Bode plot)
  - V-I characteristic curves
- **Differential Equations**:
  ```
  di/dt = (V(t) - vc - R·i) / L
  dvc/dt = i / C
  ```

### 4. Power System Analysis Tab
- **Three-Phase Power Calculations**:
  - Apparent power (S = √3 × VL × IL)
  - Real power (P = S × cos φ)
  - Reactive power (Q = S × sin φ)
- **Power Factor Correction**:
  - Calculates required capacitor bank
  - Optimizes to target power factor (0.95)
- **Current Analysis**:
  - Active and reactive current components

## Technical Implementation

### ODE Solvers

#### Euler Method
- First-order numerical integration
- Fast but less accurate
- Good for quick approximations

#### RK45 (Runge-Kutta 4th/5th Order)
- High-accuracy adaptive method
- Better stability for stiff systems
- Recommended for precise simulations

### Mathematical Models

#### DC Motor Model
```python
State variables: [current, angular_velocity]
di/dt = (V - R·i - Ke·ω) / L
dω/dt = (Kt·i - B·ω) / J
```

#### RLC Circuit Model
```python
State variables: [current, capacitor_voltage]
di/dt = (V_source(t) - vc - R·i) / L
dvc/dt = i / C
```

### GUI Features

1. **Tabbed Interface**: Organized modules for different analyses
2. **Adjustable Sliders**: Real-time parameter control
3. **Interactive Controls**: Start, Stop, Reset buttons
4. **Auto-scaling**: Automatic window and plot resizing
5. **Professional Visualization**: Matplotlib integration with multiple plots
6. **Status Bar**: Real-time feedback on operations

## Usage Instructions

### Installation Requirements
```bash
pip install numpy matplotlib
```

Note: tkinter is usually included with Python installations

### Running the Application
```bash
python3 feeder_cable_advanced_simulator.py
```

### Using the Feeder Cable Optimizer

1. Navigate to "Feeder Cable Optimizer" tab
2. Enter system parameters (default values from problem are pre-filled)
3. Click "Calculate Optimal Cross-Section"
4. Review detailed results and cost optimization graph
5. Adjust parameters to explore different scenarios

### Using Dynamic Simulations

**DC Motor:**
1. Go to "DC Motor Dynamics" tab
2. Adjust motor parameters using sliders
3. Select ODE solver (RK45 recommended)
4. Click "Start" to run simulation
5. Observe current, speed, torque, and power response
6. Click "Reset" to clear plots

**RLC Circuit:**
1. Go to "RLC Circuit Analysis" tab
2. Adjust R, L, C values and source parameters
3. Select ODE solver
4. Click "Start" to simulate
5. View transient response and frequency characteristics
6. Analyze resonance behavior

### Power System Analysis

1. Navigate to "Power System Analysis" tab
2. Enter line voltage, current, and power factor
3. Click "Calculate Power"
4. Review three-phase power calculations
5. Check power factor correction recommendations

## Advanced Features

### Real-time Parameter Adjustment
- Sliders update values instantly
- See parameter effects immediately
- No need to re-enter values

### Multiple Visualization Modes
- Time-domain plots
- Frequency-domain analysis
- Phase portraits
- Cost optimization curves

### Solver Comparison
- Compare Euler vs RK45 methods
- Understand numerical accuracy
- Educational tool for learning ODE methods

## Practical Applications

1. **Cable Design**: Economic conductor sizing for power distribution
2. **Motor Control**: Understanding DC motor startup and steady-state behavior
3. **Filter Design**: RLC circuit tuning and resonance analysis
4. **Power Quality**: Three-phase system analysis and PF correction
5. **Education**: Visual learning tool for electrical engineering concepts

## Code Structure

```
feeder_cable_advanced_simulator.py
├── ODESolver class
│   ├── euler() - Euler integration method
│   └── rk45() - Runge-Kutta 4/5 method
├── FeederCableOptimizer class
│   ├── calculate_optimal_cross_section()
│   └── cost_vs_area()
├── ElectricalMachineSimulator class
│   ├── dc_motor_model()
│   ├── rlc_circuit_model()
│   └── transformer_inrush_model()
└── AdvancedElectricalEngineeringGUI class
    ├── Feeder Cable Tab
    ├── DC Motor Tab
    ├── RLC Circuit Tab
    └── Power System Tab
```

## Mathematical Background

### Economic Conductor Sizing

The Kelvin's Law states that the most economical conductor size is achieved when:
**Annual cost of energy losses = Annual fixed charges on capital**

This is derived by setting dC/dA = 0, where:
- C = Total annual cost
- A = Cross-sectional area

### Dynamic System Modeling

Electrical systems are modeled using first-order differential equations:
- State-space representation
- Time-domain analysis
- Numerical integration for non-linear systems

## Troubleshooting

**Issue**: Graphs not displaying
- **Solution**: Ensure matplotlib backend is properly configured

**Issue**: Simulation too slow
- **Solution**: Reduce time steps or use Euler method for faster computation

**Issue**: Numerical instability
- **Solution**: Use RK45 solver or reduce step size

## Future Enhancements

- [ ] Transformer modeling with saturation curves
- [ ] Induction motor slip-torque characteristics
- [ ] Transmission line modeling (ABCD parameters)
- [ ] Short circuit analysis
- [ ] Load flow calculations
- [ ] Harmonic analysis

## Author Notes

This application combines theoretical electrical engineering principles with practical numerical methods. It's designed for:
- Students learning power systems and electrical machines
- Engineers performing quick calculations
- Researchers exploring system dynamics
- Anyone interested in electrical engineering simulation

## License

Open source - Educational and commercial use permitted
