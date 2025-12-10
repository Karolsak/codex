# Copilot Instructions for Codex Repository

This repository contains electrical engineering simulation tools written in Python with Tkinter GUIs.

## Project Overview

This is a collection of electrical engineering calculators and simulators for:
- **Electrical Tariff Calculator**: Power system tariff calculations with cost analysis
- **Generator Dynamics Simulator**: Real-time power system generator dynamics using swing equations
- **RLC Circuit Dynamics**: Series RLC circuit behavior simulation
- **Motor Analysis**: Economic analysis for motor replacement decisions
- **Feeder Cable Simulator**: Advanced feeder cable calculations
- **Diversity Factor Calculator**: Load diversity calculations

## Repository Structure

```
codex/
├── electrical_engineering_simulator.py  # Main tariff and dynamics simulator
├── motor_analysis_simulator.py          # Motor economic analysis with GUI
├── motor_analysis_core.py               # Core motor analysis logic
├── feeder_cable_advanced_simulator.py   # Feeder cable calculations
├── diversity_factor_calculator.py       # Diversity factor tool
├── test_core_functionality.py           # Core functionality tests
├── test_motor_simulator.py              # Motor simulator tests
├── requirements.txt                     # Python dependencies
└── README_*.md                          # Documentation files
```

## Technology Stack

- **Language**: Python 3.x
- **GUI Framework**: Tkinter (Python standard library)
- **Scientific Computing**: NumPy, Matplotlib
- **ODE Solvers**: Custom implementations (Euler, RK45)

## Build and Run Instructions

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Applications

```bash
# Run electrical engineering simulator
python3 electrical_engineering_simulator.py

# Run motor analysis simulator
python3 motor_analysis_simulator.py

# Run feeder cable simulator
python3 feeder_cable_advanced_simulator.py

# Run diversity factor calculator
python3 diversity_factor_calculator.py
```

### Testing

```bash
# Run core functionality tests
python3 test_core_functionality.py

# Run motor simulator tests
python3 test_motor_simulator.py
```

Note: Tests are currently non-GUI unit tests for core logic only.

## Code Style and Conventions

### Python Style

- Follow PEP 8 style guidelines
- Use descriptive variable names that reflect electrical engineering terminology
- Use docstrings for classes and non-trivial functions
- Keep functions focused and modular

### Variable Naming Conventions

- Use electrical engineering standard abbreviations when appropriate:
  - `kW` for kilowatts (power)
  - `kWh` for kilowatt-hours (energy)
  - `MW` for megawatts
  - `H` for inertia constant
  - `D` for damping coefficient
  - `Pm`, `Pe` for mechanical/electrical power
  - `R`, `L`, `C` for resistance, inductance, capacitance
  - `V` or `Vs` for voltage
- Use `_` for internal/private methods
- Use clear, full names for user-facing labels and GUI elements

### GUI Development

- Use Tkinter with ttk for modern widget styling
- Organize GUI code in classes
- Use frames to group related controls
- Implement proper event handling and validation
- Set reasonable default values based on typical electrical engineering scenarios
- Include units in labels (e.g., "Voltage (V)", "Current (A)")

### Mathematical Modeling

- Document equations in docstrings or comments using standard notation
- Implement ODE solvers as separate methods/classes for reusability
- Use NumPy arrays for numerical computations
- Validate input parameters to avoid division by zero or invalid states

### Error Handling

- Use try-except blocks for user input validation
- Show user-friendly error messages via messagebox
- Validate numerical inputs before calculations
- Handle edge cases (zero values, negative values where not allowed)

## Dependencies

### Required

- `numpy>=1.20.0` - Numerical computations
- `matplotlib>=3.3.0` - Plotting and visualization

### Standard Library (No Installation Required)

- `tkinter` - GUI framework (included with Python)
- `threading` - For real-time simulations
- `time` - Timing utilities

### Adding New Dependencies

When adding dependencies:
1. Update `requirements.txt` with version constraints
2. Test compatibility with Python 3.7+
3. Prefer standard library when possible to minimize dependencies
4. Document the purpose of each new dependency

## Domain-Specific Guidelines

### Electrical Engineering Concepts

When working with electrical engineering calculations:

1. **Units**: Always be explicit about units in variable names, labels, and documentation
2. **Precision**: Use appropriate decimal precision for electrical quantities
3. **Validation**: Validate that electrical parameters are physically realistic
4. **Standards**: Reference relevant electrical engineering standards when applicable

### Common Calculations

- **Power**: P = V × I (for DC or AC power factor = 1)
- **Energy**: E = P × t
- **Load Factor**: Average Demand / Maximum Demand
- **Tariff**: Fixed charges + Running charges based on consumption
- **Efficiency**: Output Power / Input Power

### ODE Solving

For differential equation problems:
- Use RK45 for better accuracy in production
- Euler method is acceptable for simple demonstrations
- Always specify initial conditions clearly
- Set appropriate time steps (`dt`) for stability

## Testing Guidelines

### What to Test

- Core calculation logic (separate from GUI)
- Edge cases and boundary conditions
- Physical validity of results
- Numerical stability of solvers

### What Not to Test

- Tkinter GUI components (manual testing preferred)
- Matplotlib plotting (visual verification)
- User interaction flows

### Test Structure

- Use descriptive test names that indicate what is being tested
- Test one concept per test when possible
- Include expected values based on hand calculations or references
- Print clear output showing test results

## Common Tasks

### Adding a New Calculator/Simulator

1. Create a new Python file with descriptive name
2. Implement core calculation logic as a separate class
3. Create GUI class if needed (follow existing Tkinter patterns)
4. Add README documentation explaining the tool
5. Update main README if it's a major feature
6. Create basic tests for core logic

### Modifying Calculations

1. Document the change and reference any standards/formulas
2. Update docstrings to reflect changes
3. Test with known values to verify correctness
4. Update relevant README files

### GUI Improvements

1. Maintain consistent layout with existing tools
2. Use ttk widgets for modern appearance
3. Include proper labels with units
4. Add input validation
5. Show results in organized, readable format

## Best Practices for This Repository

1. **Electrical Accuracy**: Verify calculations against textbooks or standards
2. **User Experience**: Make tools intuitive for electrical engineers
3. **Documentation**: Include formulas and references in code and README files
4. **Modularity**: Keep calculation logic separate from GUI code
5. **Testing**: Write tests for core calculation functions
6. **Educational Value**: Code should be readable and educational for students

## Helpful Context

This repository serves both as practical tools for electrical engineers and educational examples for students learning:
- Power system analysis
- Circuit theory
- Economic analysis for electrical systems
- Numerical methods (ODE solving)
- GUI development with Python

When making changes, consider both the practical utility and educational clarity of the code.
