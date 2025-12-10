# GitHub Copilot Instructions for Codex Repository

## Project Overview

This repository contains electrical engineering simulation and analysis applications written in Python. The applications provide GUI-based tools for:

- **Motor Analysis**: Economic analysis and dynamic simulation of induction motors
- **Electrical Engineering Simulator**: Tariff calculations, generator dynamics, and RLC circuit analysis
- **Feeder Cable Simulator**: Advanced feeder cable calculations
- **Diversity Factor Calculator**: Power system diversity factor calculations

## Repository Structure

```
codex/
├── motor_analysis_simulator.py       # Full GUI application for motor analysis
├── motor_analysis_core.py            # Core motor analysis logic (no GUI)
├── electrical_engineering_simulator.py   # Multi-tab electrical engineering tools
├── feeder_cable_advanced_simulator.py   # Feeder cable calculations
├── diversity_factor_calculator.py    # Diversity factor calculations
├── test_core_functionality.py        # Core functionality tests
├── test_motor_simulator.py           # Motor simulator tests
├── requirements.txt                  # Python dependencies
├── README_MOTOR_SIMULATOR.md         # Motor simulator documentation
├── README_ELECTRICAL_SIMULATOR.md    # Electrical simulator documentation
└── FEEDER_CABLE_README.md           # Feeder cable documentation
```

## Technology Stack

- **Language**: Python 3.7+
- **GUI Framework**: Tkinter (standard library)
- **Numerical Computing**: NumPy (>=1.20.0)
- **Visualization**: Matplotlib (>=3.3.0)
- **ODE Solvers**: RK45 (Runge-Kutta 4th/5th order) and Euler method

## Coding Standards and Conventions

### Python Style
- Follow PEP 8 style guide for Python code
- Use descriptive variable names that reflect electrical engineering concepts
- Use type hints where appropriate for better code clarity
- Document complex mathematical formulas in comments

### Naming Conventions
- **Classes**: PascalCase (e.g., `MotorEconomicAnalysis`, `MotorDynamicSimulator`)
- **Functions/Methods**: snake_case (e.g., `analyze_motors`, `simulate_step`)
- **Constants**: UPPER_SNAKE_CASE (e.g., for physical constants)
- **Variables**: snake_case (e.g., `motor_power`, `load_torque`)

### GUI Development
- Use Tkinter for GUI applications (already standard in Python)
- Organize GUI code with clear separation between:
  - Economic analysis modules
  - Dynamic simulation modules
  - Visualization components
- Use `ttk` widgets for modern appearance where applicable
- Implement responsive layouts with proper grid/pack geometry managers
- Window sizes should be configurable and responsive

### Mathematical and Engineering Code
- Include clear comments explaining electrical engineering formulas
- Use physically meaningful variable names (e.g., `efficiency`, `power_factor`, `torque`)
- Document units in comments (e.g., kW, RPM, N.m, A, V)
- Validate input ranges for physical parameters
- Handle edge cases in calculations (division by zero, negative values where inappropriate)

### Documentation
- All modules should have docstrings explaining their purpose
- Complex functions should have detailed docstrings with:
  - Parameters and their units
  - Return values and their units
  - Example usage when applicable
- Update README files when adding new features or changing behavior

## Dependencies and Installation

### Required Packages
```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:
- numpy>=1.20.0
- matplotlib>=3.3.0

### Platform-Specific Requirements
- **Linux**: `sudo apt-get install python3-tk`
- **macOS**: Tkinter usually included with Python installation from python.org, or install via Homebrew: `brew install python-tk@3.11` (replace 3.11 with your Python version)
- **Windows**: Tkinter usually included with Python installation

## Building and Testing

### Running Applications
```bash
# Motor analysis with GUI
python3 motor_analysis_simulator.py

# Electrical engineering simulator
python3 electrical_engineering_simulator.py

# Feeder cable simulator
python3 feeder_cable_advanced_simulator.py

# Diversity factor calculator
python3 diversity_factor_calculator.py
```

### Running Tests
```bash
# Test core functionality
python3 test_core_functionality.py

# Test motor simulator
python3 test_motor_simulator.py
```

### Code Validation
```bash
# Check Python syntax
python3 -m py_compile <filename>.py

# Run all tests
python3 test_core_functionality.py && python3 test_motor_simulator.py
```

## Domain-Specific Knowledge

### Electrical Engineering Concepts
This repository deals with:
- **Motor Analysis**: Induction motor dynamics, torque-speed characteristics, efficiency calculations
- **Power Systems**: Generator dynamics (swing equation), tariff calculations, load factors
- **Circuit Theory**: RLC circuits, impedance, transient analysis
- **Economic Analysis**: Present worth calculations, life cycle costing, energy cost analysis

### Common Physical Parameters
- **Power**: Typically in kW or MW
- **Voltage**: Volts (V)
- **Current**: Amperes (A)
- **Torque**: Newton-meters (N.m)
- **Speed**: RPM (revolutions per minute)
- **Efficiency**: Percentage (0-100%) or per-unit (0-1.0)
- **Energy**: kWh (kilowatt-hours)
- **Frequency**: Hz (50 or 60 Hz for power systems)

### Mathematical Models
- **ODE Solvers**: RK45 for accuracy, Euler for speed and educational purposes
- **Motor Dynamics**: d-q reference frame modeling
- **Generator Dynamics**: Swing equation for stability analysis
- **Economic Models**: Present worth analysis with discount rates

## Common Tasks and Guidelines

### Adding New Features
1. Maintain separation between core logic and GUI code (follow the pattern of `motor_analysis_core.py` vs `motor_analysis_simulator.py`)
2. Add appropriate tests in the test files
3. Update relevant README files with new feature documentation
4. Ensure new features work with existing GUI layout and don't break responsive design

### Modifying Simulations
1. Validate mathematical correctness of formulas
2. Ensure units are consistent throughout calculations
3. Test with edge cases (zero values, maximum values, negative values where applicable)
4. Update visualization code if state variables change

### Bug Fixes
1. Identify the root cause before making changes
2. Test the fix with multiple scenarios
3. Ensure the fix doesn't break existing functionality
4. Update tests if the fix changes expected behavior

### Performance Optimization
1. NumPy operations are generally optimized; use vectorization when possible
2. For real-time GUI updates, consider update frequency (don't update too frequently)
3. Profile code before optimizing to identify actual bottlenecks

## Error Handling and Validation
- Validate all user inputs from GUI forms (check ranges, types, and reasonable values)
- Ensure numerical stability in calculations (avoid division by zero, check for NaN/infinity)
- Handle exceptions gracefully with user-friendly error messages
- Validate physical parameter ranges (e.g., efficiency must be 0-100%, power must be positive)

## Git Practices
- Write clear, descriptive commit messages
- Keep commits focused on single changes
- Test code before committing
- Update documentation in the same commit as code changes

## Common Patterns in This Repository

### GUI Application Structure
```python
class ApplicationGUI:
    def __init__(self, root):
        self.root = root
        self.setup_gui()
    
    def setup_gui(self):
        # Create tabs, frames, widgets
        pass
    
    def calculate(self):
        # Perform calculations
        # Update GUI with results
        pass
```

### Simulation Loop
```python
class Simulator:
    def simulate_step(self, method='rk45'):
        # Calculate derivatives
        # Update state using ODE solver
        # Return current state
        pass
```

### Economic Analysis Pattern
```python
class EconomicAnalysis:
    def analyze(self):
        # Calculate present worth
        # Compare alternatives
        # Return recommendation
        pass
```

## Additional Notes

- This is primarily an educational and professional tool for electrical engineering applications
- Code should be readable and maintainable for students and engineers
- Prefer clarity over optimization unless performance is critical
- Mathematical accuracy is paramount - always verify formulas against electrical engineering references
- GUI should be intuitive for users familiar with electrical engineering concepts
