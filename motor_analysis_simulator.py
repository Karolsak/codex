"""
Advanced Motor Replacement Analysis and Dynamic Simulation System
Combines economic analysis with real-time motor dynamics simulation
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from datetime import datetime


class MotorEconomicAnalysis:
    """Economic analysis for motor replacement decision"""

    def __init__(self):
        self.reset_parameters()

    def reset_parameters(self):
        """Reset to default parameters"""
        # Motor A parameters
        self.motor_a_cost = 6000
        self.motor_a_eta_full = 0.90
        self.motor_a_eta_half = 0.86
        self.motor_a_maintenance = 420

        # Motor B parameters
        self.motor_b_cost = 4000
        self.motor_b_eta_full = 0.85
        self.motor_b_eta_half = 0.82
        self.motor_b_maintenance = 240

        # Common parameters
        self.motor_power = 22.5  # kW
        self.life_years = 20
        self.salvage_rate = 0.10
        self.interest_rate = 0.05
        self.full_load_time = 0.25
        self.half_load_time = 0.75
        self.energy_rate = 0.10  # Rs per kWh
        self.operating_hours = 8760  # hours per year

    def calculate_annual_energy_cost(self, eta_full, eta_half):
        """Calculate annual energy consumption cost"""
        # Energy at full load
        full_load_hours = self.operating_hours * self.full_load_time
        energy_full = (self.motor_power / eta_full) * full_load_hours

        # Energy at half load
        half_load_hours = self.operating_hours * self.half_load_time
        energy_half = (self.motor_power * 0.5 / eta_half) * half_load_hours

        # Total annual energy cost
        total_energy = energy_full + energy_half
        annual_cost = total_energy * self.energy_rate

        return annual_cost, total_energy

    def calculate_present_worth(self, initial_cost, annual_cost, maintenance, salvage_value):
        """Calculate present worth of all costs"""
        # Present worth of annual costs
        n = self.life_years
        i = self.interest_rate

        # Uniform series present worth factor
        pw_factor = ((1 + i)**n - 1) / (i * (1 + i)**n)

        # Present worth of operating costs
        pw_operating = (annual_cost + maintenance) * pw_factor

        # Present worth of salvage value (negative cost)
        pw_salvage = salvage_value / ((1 + i)**n)

        # Total present worth
        total_pw = initial_cost + pw_operating - pw_salvage

        return total_pw, pw_operating, pw_salvage

    def analyze_motors(self):
        """Compare both motors and return detailed analysis"""
        results = {}

        # Motor A analysis
        salvage_a = self.motor_a_cost * self.salvage_rate
        annual_energy_a, total_energy_a = self.calculate_annual_energy_cost(
            self.motor_a_eta_full, self.motor_a_eta_half
        )
        pw_a, pw_op_a, pw_salvage_a = self.calculate_present_worth(
            self.motor_a_cost, annual_energy_a, self.motor_a_maintenance, salvage_a
        )

        results['Motor A'] = {
            'initial_cost': self.motor_a_cost,
            'annual_energy_cost': annual_energy_a,
            'annual_energy_kwh': total_energy_a,
            'annual_maintenance': self.motor_a_maintenance,
            'total_annual_cost': annual_energy_a + self.motor_a_maintenance,
            'salvage_value': salvage_a,
            'present_worth': pw_a,
            'pw_operating': pw_op_a,
            'pw_salvage': pw_salvage_a
        }

        # Motor B analysis
        salvage_b = self.motor_b_cost * self.salvage_rate
        annual_energy_b, total_energy_b = self.calculate_annual_energy_cost(
            self.motor_b_eta_full, self.motor_b_eta_half
        )
        pw_b, pw_op_b, pw_salvage_b = self.calculate_present_worth(
            self.motor_b_cost, annual_energy_b, self.motor_b_maintenance, salvage_b
        )

        results['Motor B'] = {
            'initial_cost': self.motor_b_cost,
            'annual_energy_cost': annual_energy_b,
            'annual_energy_kwh': total_energy_b,
            'annual_maintenance': self.motor_b_maintenance,
            'total_annual_cost': annual_energy_b + self.motor_b_maintenance,
            'salvage_value': salvage_b,
            'present_worth': pw_b,
            'pw_operating': pw_op_b,
            'pw_salvage': pw_salvage_b
        }

        # Recommendation
        if pw_a < pw_b:
            results['recommendation'] = 'Motor A'
            results['savings'] = pw_b - pw_a
        else:
            results['recommendation'] = 'Motor B'
            results['savings'] = pw_a - pw_b

        return results


class MotorDynamicSimulator:
    """Dynamic motor simulation using differential equations"""

    def __init__(self):
        self.reset_parameters()
        self.simulation_running = False
        self.simulation_thread = None

    def reset_parameters(self):
        """Reset simulation parameters"""
        # Motor parameters
        self.rated_power = 22.5e3  # W
        self.rated_voltage = 415  # V
        self.rated_speed = 1450  # RPM
        self.poles = 4
        self.Rs = 0.5  # Stator resistance (Ohms)
        self.Rr = 0.3  # Rotor resistance (Ohms)
        self.Ls = 0.05  # Stator inductance (H)
        self.Lr = 0.05  # Rotor inductance (H)
        self.Lm = 0.045  # Mutual inductance (H)
        self.J = 0.5  # Moment of inertia (kg.m^2)
        self.B = 0.01  # Friction coefficient

        # Simulation parameters
        self.dt = 0.001  # Time step (s)
        self.load_torque = 50  # N.m
        self.voltage_amplitude = 415 * np.sqrt(2/3)

        # State variables
        self.reset_state()

    def reset_state(self):
        """Reset state variables"""
        self.time = 0
        self.omega = 0  # Angular velocity (rad/s)
        self.theta = 0  # Rotor position (rad)
        self.ids = 0  # d-axis stator current
        self.iqs = 0  # q-axis stator current
        self.idr = 0  # d-axis rotor current
        self.iqr = 0  # q-axis rotor current

        # Data storage
        self.time_data = []
        self.speed_data = []
        self.torque_data = []
        self.current_data = []
        self.power_data = []

    def motor_equations_euler(self, state, voltage, load_torque):
        """
        Motor differential equations using Euler method
        state = [ids, iqs, idr, iqr, omega, theta]
        """
        ids, iqs, idr, iqr, omega, theta = state

        # Synchronous speed
        omega_s = 2 * np.pi * 50  # 50 Hz
        omega_slip = omega_s - omega

        # Voltage equations (simplified induction motor model)
        vds = voltage * np.cos(omega_s * self.time)
        vqs = voltage * np.sin(omega_s * self.time)

        # Current derivatives
        L_sigma_s = self.Ls - self.Lm**2 / self.Lr
        L_sigma_r = self.Lr - self.Lm**2 / self.Ls

        dids = (vds - self.Rs * ids + omega_s * self.Ls * iqs) / L_sigma_s
        diqs = (vqs - self.Rs * iqs - omega_s * self.Ls * ids) / L_sigma_s
        didr = (-self.Rr * idr + omega_slip * self.Lr * iqr) / L_sigma_r
        diqr = (-self.Rr * iqr - omega_slip * self.Lr * idr) / L_sigma_r

        # Electromagnetic torque
        Te = 1.5 * self.poles * self.Lm * (iqs * idr - ids * iqr)

        # Mechanical equation
        domega = (Te - load_torque - self.B * omega) / self.J
        dtheta = omega

        return np.array([dids, diqs, didr, diqr, domega, dtheta]), Te

    def motor_equations_rk45(self, state, voltage, load_torque, dt):
        """
        Motor differential equations using RK45 method
        """
        def derivatives(s, t_local):
            deriv, _ = self.motor_equations_euler(s, voltage, load_torque)
            return deriv

        # RK45 coefficients
        k1 = derivatives(state, self.time)
        k2 = derivatives(state + dt * k1 / 2, self.time + dt / 2)
        k3 = derivatives(state + dt * k2 / 2, self.time + dt / 2)
        k4 = derivatives(state + dt * k3, self.time + dt)

        # Weighted average
        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        # Calculate torque at new state
        _, Te = self.motor_equations_euler(new_state, voltage, load_torque)

        return new_state, Te

    def simulate_step(self, method='rk45'):
        """Perform one simulation step"""
        state = np.array([self.ids, self.iqs, self.idr, self.iqr, self.omega, self.theta])

        if method == 'euler':
            derivs, Te = self.motor_equations_euler(state, self.voltage_amplitude, self.load_torque)
            new_state = state + self.dt * derivs
        else:  # rk45
            new_state, Te = self.motor_equations_rk45(state, self.voltage_amplitude, self.load_torque, self.dt)

        # Update state
        self.ids, self.iqs, self.idr, self.iqr, self.omega, self.theta = new_state
        self.time += self.dt

        # Calculate outputs
        speed_rpm = self.omega * 60 / (2 * np.pi)
        current = np.sqrt(self.ids**2 + self.iqs**2)
        power = Te * self.omega

        # Store data (downsample for efficiency)
        if len(self.time_data) == 0 or self.time - self.time_data[-1] >= 0.01:
            self.time_data.append(self.time)
            self.speed_data.append(speed_rpm)
            self.torque_data.append(Te)
            self.current_data.append(current)
            self.power_data.append(power / 1000)  # kW

        return speed_rpm, Te, current, power


class MotorAnalysisGUI:
    """Main GUI application"""

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Motor Analysis & Simulation System")
        self.root.geometry("1400x900")

        # Initialize analysis modules
        self.economic_analysis = MotorEconomicAnalysis()
        self.dynamic_sim = MotorDynamicSimulator()

        # Simulation control
        self.sim_running = False
        self.sim_method = 'rk45'

        # Setup GUI
        self.setup_gui()

        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)

    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 1: Economic Analysis
        self.tab_economic = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_economic, text='Economic Analysis')
        self.setup_economic_tab()

        # Tab 2: Dynamic Simulation
        self.tab_simulation = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_simulation, text='Dynamic Simulation')
        self.setup_simulation_tab()

        # Tab 3: Comparison & Results
        self.tab_comparison = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_comparison, text='Comprehensive Results')
        self.setup_comparison_tab()

    def setup_economic_tab(self):
        """Setup economic analysis tab"""
        # Left panel - Input parameters
        left_frame = ttk.LabelFrame(self.tab_economic, text="Input Parameters", padding=10)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Motor A parameters
        ttk.Label(left_frame, text="MOTOR A PARAMETERS", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Label(left_frame, text="Initial Cost (Rs):").grid(row=1, column=0, sticky='w')
        self.entry_a_cost = ttk.Entry(left_frame, width=15)
        self.entry_a_cost.insert(0, "6000")
        self.entry_a_cost.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Efficiency @ Full Load (%):").grid(row=2, column=0, sticky='w')
        self.entry_a_eta_full = ttk.Entry(left_frame, width=15)
        self.entry_a_eta_full.insert(0, "90")
        self.entry_a_eta_full.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Efficiency @ Half Load (%):").grid(row=3, column=0, sticky='w')
        self.entry_a_eta_half = ttk.Entry(left_frame, width=15)
        self.entry_a_eta_half.insert(0, "86")
        self.entry_a_eta_half.grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Annual Maintenance (Rs):").grid(row=4, column=0, sticky='w')
        self.entry_a_maintenance = ttk.Entry(left_frame, width=15)
        self.entry_a_maintenance.insert(0, "420")
        self.entry_a_maintenance.grid(row=4, column=1, padx=5, pady=2)

        # Motor B parameters
        ttk.Label(left_frame, text="MOTOR B PARAMETERS", font=('Arial', 10, 'bold')).grid(row=5, column=0, columnspan=2, pady=(15, 5))

        ttk.Label(left_frame, text="Initial Cost (Rs):").grid(row=6, column=0, sticky='w')
        self.entry_b_cost = ttk.Entry(left_frame, width=15)
        self.entry_b_cost.insert(0, "4000")
        self.entry_b_cost.grid(row=6, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Efficiency @ Full Load (%):").grid(row=7, column=0, sticky='w')
        self.entry_b_eta_full = ttk.Entry(left_frame, width=15)
        self.entry_b_eta_full.insert(0, "85")
        self.entry_b_eta_full.grid(row=7, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Efficiency @ Half Load (%):").grid(row=8, column=0, sticky='w')
        self.entry_b_eta_half = ttk.Entry(left_frame, width=15)
        self.entry_b_eta_half.insert(0, "82")
        self.entry_b_eta_half.grid(row=8, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Annual Maintenance (Rs):").grid(row=9, column=0, sticky='w')
        self.entry_b_maintenance = ttk.Entry(left_frame, width=15)
        self.entry_b_maintenance.insert(0, "240")
        self.entry_b_maintenance.grid(row=9, column=1, padx=5, pady=2)

        # Common parameters
        ttk.Label(left_frame, text="COMMON PARAMETERS", font=('Arial', 10, 'bold')).grid(row=10, column=0, columnspan=2, pady=(15, 5))

        ttk.Label(left_frame, text="Motor Power (kW):").grid(row=11, column=0, sticky='w')
        self.entry_power = ttk.Entry(left_frame, width=15)
        self.entry_power.insert(0, "22.5")
        self.entry_power.grid(row=11, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Life (years):").grid(row=12, column=0, sticky='w')
        self.entry_life = ttk.Entry(left_frame, width=15)
        self.entry_life.insert(0, "20")
        self.entry_life.grid(row=12, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Interest Rate (%):").grid(row=13, column=0, sticky='w')
        self.entry_interest = ttk.Entry(left_frame, width=15)
        self.entry_interest.insert(0, "5")
        self.entry_interest.grid(row=13, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Energy Rate (Rs/kWh):").grid(row=14, column=0, sticky='w')
        self.entry_energy_rate = ttk.Entry(left_frame, width=15)
        self.entry_energy_rate.insert(0, "0.10")
        self.entry_energy_rate.grid(row=14, column=1, padx=5, pady=2)

        ttk.Label(left_frame, text="Full Load Time (%):").grid(row=15, column=0, sticky='w')
        self.entry_full_load = ttk.Entry(left_frame, width=15)
        self.entry_full_load.insert(0, "25")
        self.entry_full_load.grid(row=15, column=1, padx=5, pady=2)

        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=16, column=0, columnspan=2, pady=10)

        ttk.Button(btn_frame, text="Calculate", command=self.calculate_economic).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset_economic).pack(side='left', padx=5)

        # Right panel - Results
        right_frame = ttk.LabelFrame(self.tab_economic, text="Analysis Results", padding=10)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        # Results text area
        self.results_text = tk.Text(right_frame, width=60, height=30, font=('Courier', 9))
        self.results_text.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(right_frame, command=self.results_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Configure grid weights
        self.tab_economic.columnconfigure(1, weight=1)
        self.tab_economic.rowconfigure(0, weight=1)

    def setup_simulation_tab(self):
        """Setup dynamic simulation tab"""
        # Control panel
        control_frame = ttk.LabelFrame(self.tab_simulation, text="Simulation Controls", padding=10)
        control_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5, columnspan=2)

        # Solver method selection
        ttk.Label(control_frame, text="ODE Solver:").grid(row=0, column=0, padx=5)
        self.solver_var = tk.StringVar(value='rk45')
        ttk.Radiobutton(control_frame, text="RK45 (Runge-Kutta)", variable=self.solver_var,
                       value='rk45').grid(row=0, column=1, padx=5)
        ttk.Radiobutton(control_frame, text="Euler Method", variable=self.solver_var,
                       value='euler').grid(row=0, column=2, padx=5)

        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=0, column=3, padx=20)

        self.btn_start = ttk.Button(btn_frame, text="Start", command=self.start_simulation)
        self.btn_start.pack(side='left', padx=3)

        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self.stop_simulation, state='disabled')
        self.btn_stop.pack(side='left', padx=3)

        self.btn_reset = ttk.Button(btn_frame, text="Reset", command=self.reset_simulation)
        self.btn_reset.pack(side='left', padx=3)

        # Parameters panel
        param_frame = ttk.LabelFrame(self.tab_simulation, text="Motor Parameters", padding=10)
        param_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        # Load torque slider
        ttk.Label(param_frame, text="Load Torque (N.m):").grid(row=0, column=0, sticky='w', pady=5)
        self.load_torque_var = tk.DoubleVar(value=50)
        self.load_torque_scale = ttk.Scale(param_frame, from_=0, to=200, orient='horizontal',
                                          variable=self.load_torque_var, command=self.update_load_torque)
        self.load_torque_scale.grid(row=0, column=1, sticky='ew', padx=5)
        self.load_torque_label = ttk.Label(param_frame, text="50.0")
        self.load_torque_label.grid(row=0, column=2, padx=5)

        # Voltage slider
        ttk.Label(param_frame, text="Voltage (V):").grid(row=1, column=0, sticky='w', pady=5)
        self.voltage_var = tk.DoubleVar(value=415)
        self.voltage_scale = ttk.Scale(param_frame, from_=0, to=500, orient='horizontal',
                                      variable=self.voltage_var, command=self.update_voltage)
        self.voltage_scale.grid(row=1, column=1, sticky='ew', padx=5)
        self.voltage_label = ttk.Label(param_frame, text="415.0")
        self.voltage_label.grid(row=1, column=2, padx=5)

        # Inertia slider
        ttk.Label(param_frame, text="Inertia (kg.m²):").grid(row=2, column=0, sticky='w', pady=5)
        self.inertia_var = tk.DoubleVar(value=0.5)
        self.inertia_scale = ttk.Scale(param_frame, from_=0.1, to=2.0, orient='horizontal',
                                      variable=self.inertia_var, command=self.update_inertia)
        self.inertia_scale.grid(row=2, column=1, sticky='ew', padx=5)
        self.inertia_label = ttk.Label(param_frame, text="0.50")
        self.inertia_label.grid(row=2, column=2, padx=5)

        param_frame.columnconfigure(1, weight=1)

        # Real-time display
        display_frame = ttk.LabelFrame(self.tab_simulation, text="Real-time Data", padding=10)
        display_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

        self.display_text = tk.Text(display_frame, width=40, height=10, font=('Courier', 10))
        self.display_text.pack(fill='both', expand=True)

        # Visualization panel
        viz_frame = ttk.LabelFrame(self.tab_simulation, text="Dynamic Visualization", padding=5)
        viz_frame.grid(row=1, column=1, rowspan=2, sticky='nsew', padx=5, pady=5)

        # Create matplotlib figure
        self.sim_fig = Figure(figsize=(8, 8), dpi=100)
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, master=viz_frame)
        self.sim_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Create subplots
        self.ax_speed = self.sim_fig.add_subplot(411)
        self.ax_torque = self.sim_fig.add_subplot(412)
        self.ax_current = self.sim_fig.add_subplot(413)
        self.ax_power = self.sim_fig.add_subplot(414)

        self.sim_fig.tight_layout()

        # Configure grid weights
        self.tab_simulation.columnconfigure(1, weight=1)
        self.tab_simulation.rowconfigure(1, weight=1)

    def setup_comparison_tab(self):
        """Setup comprehensive comparison tab"""
        # Create matplotlib figure for comparison charts
        self.comp_fig = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(self.comp_fig, master=self.tab_comparison)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        # Create subplots for comparison
        self.ax_cost_comparison = self.comp_fig.add_subplot(221)
        self.ax_energy_comparison = self.comp_fig.add_subplot(222)
        self.ax_cumulative = self.comp_fig.add_subplot(223)
        self.ax_breakdown = self.comp_fig.add_subplot(224)

        self.comp_fig.tight_layout()

        # Update button
        ttk.Button(self.tab_comparison, text="Generate Comprehensive Analysis",
                  command=self.generate_comparison).pack(pady=5)

    def update_load_torque(self, value):
        """Update load torque parameter"""
        val = float(value)
        self.load_torque_label.config(text=f"{val:.1f}")
        self.dynamic_sim.load_torque = val

    def update_voltage(self, value):
        """Update voltage parameter"""
        val = float(value)
        self.voltage_label.config(text=f"{val:.1f}")
        self.dynamic_sim.voltage_amplitude = val * np.sqrt(2/3)

    def update_inertia(self, value):
        """Update inertia parameter"""
        val = float(value)
        self.inertia_label.config(text=f"{val:.2f}")
        self.dynamic_sim.J = val

    def calculate_economic(self):
        """Perform economic analysis calculation"""
        try:
            # Update parameters from GUI
            self.economic_analysis.motor_a_cost = float(self.entry_a_cost.get())
            self.economic_analysis.motor_a_eta_full = float(self.entry_a_eta_full.get()) / 100
            self.economic_analysis.motor_a_eta_half = float(self.entry_a_eta_half.get()) / 100
            self.economic_analysis.motor_a_maintenance = float(self.entry_a_maintenance.get())

            self.economic_analysis.motor_b_cost = float(self.entry_b_cost.get())
            self.economic_analysis.motor_b_eta_full = float(self.entry_b_eta_full.get()) / 100
            self.economic_analysis.motor_b_eta_half = float(self.entry_b_eta_half.get()) / 100
            self.economic_analysis.motor_b_maintenance = float(self.entry_b_maintenance.get())

            self.economic_analysis.motor_power = float(self.entry_power.get())
            self.economic_analysis.life_years = int(self.entry_life.get())
            self.economic_analysis.interest_rate = float(self.entry_interest.get()) / 100
            self.economic_analysis.energy_rate = float(self.entry_energy_rate.get())
            self.economic_analysis.full_load_time = float(self.entry_full_load.get()) / 100
            self.economic_analysis.half_load_time = 1 - self.economic_analysis.full_load_time

            # Perform analysis
            results = self.economic_analysis.analyze_motors()

            # Display results
            self.display_economic_results(results)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input values: {str(e)}")

    def display_economic_results(self, results):
        """Display economic analysis results"""
        self.results_text.delete(1.0, tk.END)

        output = "=" * 70 + "\n"
        output += "MOTOR REPLACEMENT ECONOMIC ANALYSIS REPORT\n"
        output += "=" * 70 + "\n\n"
        output += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"Analysis Period: {self.economic_analysis.life_years} years\n"
        output += f"Interest Rate: {self.economic_analysis.interest_rate * 100:.2f}%\n"
        output += f"Operating Hours: {self.economic_analysis.operating_hours} hours/year\n\n"

        output += "-" * 70 + "\n"
        output += "MOTOR A - DETAILED ANALYSIS\n"
        output += "-" * 70 + "\n"
        motor_a = results['Motor A']
        output += f"Initial Cost:              Rs. {motor_a['initial_cost']:,.2f}\n"
        output += f"Efficiency (Full Load):    {self.economic_analysis.motor_a_eta_full * 100:.2f}%\n"
        output += f"Efficiency (Half Load):    {self.economic_analysis.motor_a_eta_half * 100:.2f}%\n"
        output += f"\nAnnual Operating Costs:\n"
        output += f"  Energy Consumption:      {motor_a['annual_energy_kwh']:,.2f} kWh\n"
        output += f"  Energy Cost:             Rs. {motor_a['annual_energy_cost']:,.2f}\n"
        output += f"  Maintenance Cost:        Rs. {motor_a['annual_maintenance']:,.2f}\n"
        output += f"  Total Annual Cost:       Rs. {motor_a['total_annual_cost']:,.2f}\n"
        output += f"\nPresent Worth Analysis:\n"
        output += f"  PW of Operating Costs:   Rs. {motor_a['pw_operating']:,.2f}\n"
        output += f"  Salvage Value:           Rs. {motor_a['salvage_value']:,.2f}\n"
        output += f"  PW of Salvage:           Rs. {motor_a['pw_salvage']:,.2f}\n"
        output += f"  TOTAL PRESENT WORTH:     Rs. {motor_a['present_worth']:,.2f}\n\n"

        output += "-" * 70 + "\n"
        output += "MOTOR B - DETAILED ANALYSIS\n"
        output += "-" * 70 + "\n"
        motor_b = results['Motor B']
        output += f"Initial Cost:              Rs. {motor_b['initial_cost']:,.2f}\n"
        output += f"Efficiency (Full Load):    {self.economic_analysis.motor_b_eta_full * 100:.2f}%\n"
        output += f"Efficiency (Half Load):    {self.economic_analysis.motor_b_eta_half * 100:.2f}%\n"
        output += f"\nAnnual Operating Costs:\n"
        output += f"  Energy Consumption:      {motor_b['annual_energy_kwh']:,.2f} kWh\n"
        output += f"  Energy Cost:             Rs. {motor_b['annual_energy_cost']:,.2f}\n"
        output += f"  Maintenance Cost:        Rs. {motor_b['annual_maintenance']:,.2f}\n"
        output += f"  Total Annual Cost:       Rs. {motor_b['total_annual_cost']:,.2f}\n"
        output += f"\nPresent Worth Analysis:\n"
        output += f"  PW of Operating Costs:   Rs. {motor_b['pw_operating']:,.2f}\n"
        output += f"  Salvage Value:           Rs. {motor_b['salvage_value']:,.2f}\n"
        output += f"  PW of Salvage:           Rs. {motor_b['pw_salvage']:,.2f}\n"
        output += f"  TOTAL PRESENT WORTH:     Rs. {motor_b['present_worth']:,.2f}\n\n"

        output += "=" * 70 + "\n"
        output += "RECOMMENDATION\n"
        output += "=" * 70 + "\n"
        output += f"\nRecommended Motor: {results['recommendation']}\n"
        output += f"Savings over {self.economic_analysis.life_years} years: Rs. {results['savings']:,.2f}\n\n"

        if results['recommendation'] == 'Motor A':
            output += "Justification: Although Motor A has a higher initial cost,\n"
            output += "its superior efficiency and lower operating costs result in\n"
            output += "lower total present worth over the motor's lifetime.\n"
        else:
            output += "Justification: Motor B's lower initial cost and maintenance\n"
            output += "costs outweigh its lower efficiency, resulting in lower\n"
            output += "total present worth over the motor's lifetime.\n"

        output += "\n" + "=" * 70 + "\n"

        self.results_text.insert(1.0, output)

    def reset_economic(self):
        """Reset economic analysis to default values"""
        self.economic_analysis.reset_parameters()

        self.entry_a_cost.delete(0, tk.END)
        self.entry_a_cost.insert(0, str(self.economic_analysis.motor_a_cost))
        self.entry_a_eta_full.delete(0, tk.END)
        self.entry_a_eta_full.insert(0, str(self.economic_analysis.motor_a_eta_full * 100))
        self.entry_a_eta_half.delete(0, tk.END)
        self.entry_a_eta_half.insert(0, str(self.economic_analysis.motor_a_eta_half * 100))
        self.entry_a_maintenance.delete(0, tk.END)
        self.entry_a_maintenance.insert(0, str(self.economic_analysis.motor_a_maintenance))

        self.entry_b_cost.delete(0, tk.END)
        self.entry_b_cost.insert(0, str(self.economic_analysis.motor_b_cost))
        self.entry_b_eta_full.delete(0, tk.END)
        self.entry_b_eta_full.insert(0, str(self.economic_analysis.motor_b_eta_full * 100))
        self.entry_b_eta_half.delete(0, tk.END)
        self.entry_b_eta_half.insert(0, str(self.economic_analysis.motor_b_eta_half * 100))
        self.entry_b_maintenance.delete(0, tk.END)
        self.entry_b_maintenance.insert(0, str(self.economic_analysis.motor_b_maintenance))

        self.entry_power.delete(0, tk.END)
        self.entry_power.insert(0, str(self.economic_analysis.motor_power))
        self.entry_life.delete(0, tk.END)
        self.entry_life.insert(0, str(self.economic_analysis.life_years))
        self.entry_interest.delete(0, tk.END)
        self.entry_interest.insert(0, str(self.economic_analysis.interest_rate * 100))
        self.entry_energy_rate.delete(0, tk.END)
        self.entry_energy_rate.insert(0, str(self.economic_analysis.energy_rate))
        self.entry_full_load.delete(0, tk.END)
        self.entry_full_load.insert(0, str(self.economic_analysis.full_load_time * 100))

        self.results_text.delete(1.0, tk.END)

    def start_simulation(self):
        """Start the dynamic simulation"""
        if not self.sim_running:
            self.sim_running = True
            self.sim_method = self.solver_var.get()
            self.btn_start.config(state='disabled')
            self.btn_stop.config(state='normal')

            # Start simulation in separate thread
            self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.sim_thread.start()

    def stop_simulation(self):
        """Stop the dynamic simulation"""
        self.sim_running = False
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')

    def reset_simulation(self):
        """Reset the simulation"""
        self.stop_simulation()
        time.sleep(0.1)  # Allow thread to stop

        self.dynamic_sim.reset_state()

        # Clear plots
        for ax in [self.ax_speed, self.ax_torque, self.ax_current, self.ax_power]:
            ax.clear()
        self.sim_canvas.draw()

        # Clear display
        self.display_text.delete(1.0, tk.END)

    def run_simulation(self):
        """Run the simulation loop"""
        max_time = 10  # Maximum simulation time in seconds

        while self.sim_running and self.dynamic_sim.time < max_time:
            # Simulate one step
            speed, torque, current, power = self.dynamic_sim.simulate_step(method=self.sim_method)

            # Update display every 100ms
            if len(self.dynamic_sim.time_data) > 0:
                self.root.after(0, self.update_simulation_display, speed, torque, current, power)

            # Control simulation speed
            time.sleep(0.001)

        self.sim_running = False
        self.root.after(0, lambda: self.btn_start.config(state='normal'))
        self.root.after(0, lambda: self.btn_stop.config(state='disabled'))

    def update_simulation_display(self, speed, torque, current, power):
        """Update simulation display and plots"""
        # Update text display
        self.display_text.delete(1.0, tk.END)
        display_str = f"Time: {self.dynamic_sim.time:.3f} s\n"
        display_str += f"Solver: {self.sim_method.upper()}\n"
        display_str += f"-" * 40 + "\n"
        display_str += f"Speed:   {speed:.2f} RPM\n"
        display_str += f"Torque:  {torque:.2f} N.m\n"
        display_str += f"Current: {current:.2f} A\n"
        display_str += f"Power:   {power/1000:.2f} kW\n"
        display_str += f"-" * 40 + "\n"
        display_str += f"Load Torque: {self.dynamic_sim.load_torque:.1f} N.m\n"
        display_str += f"Voltage: {self.voltage_var.get():.1f} V\n"
        display_str += f"Inertia: {self.dynamic_sim.J:.2f} kg.m²\n"
        self.display_text.insert(1.0, display_str)

        # Update plots
        if len(self.dynamic_sim.time_data) > 1:
            # Speed plot
            self.ax_speed.clear()
            self.ax_speed.plot(self.dynamic_sim.time_data, self.dynamic_sim.speed_data, 'b-', linewidth=2)
            self.ax_speed.set_ylabel('Speed (RPM)', fontsize=9)
            self.ax_speed.grid(True, alpha=0.3)
            self.ax_speed.set_title('Motor Speed', fontsize=10, fontweight='bold')

            # Torque plot
            self.ax_torque.clear()
            self.ax_torque.plot(self.dynamic_sim.time_data, self.dynamic_sim.torque_data, 'r-', linewidth=2)
            self.ax_torque.set_ylabel('Torque (N.m)', fontsize=9)
            self.ax_torque.grid(True, alpha=0.3)
            self.ax_torque.set_title('Electromagnetic Torque', fontsize=10, fontweight='bold')

            # Current plot
            self.ax_current.clear()
            self.ax_current.plot(self.dynamic_sim.time_data, self.dynamic_sim.current_data, 'g-', linewidth=2)
            self.ax_current.set_ylabel('Current (A)', fontsize=9)
            self.ax_current.grid(True, alpha=0.3)
            self.ax_current.set_title('Stator Current', fontsize=10, fontweight='bold')

            # Power plot
            self.ax_power.clear()
            self.ax_power.plot(self.dynamic_sim.time_data, self.dynamic_sim.power_data, 'm-', linewidth=2)
            self.ax_power.set_ylabel('Power (kW)', fontsize=9)
            self.ax_power.set_xlabel('Time (s)', fontsize=9)
            self.ax_power.grid(True, alpha=0.3)
            self.ax_power.set_title('Output Power', fontsize=10, fontweight='bold')

            self.sim_fig.tight_layout()
            self.sim_canvas.draw()

    def generate_comparison(self):
        """Generate comprehensive comparison charts"""
        try:
            # First run economic analysis
            self.calculate_economic()
            results = self.economic_analysis.analyze_motors()

            # Clear previous plots
            for ax in [self.ax_cost_comparison, self.ax_energy_comparison,
                      self.ax_cumulative, self.ax_breakdown]:
                ax.clear()

            # Plot 1: Cost Comparison
            categories = ['Initial\nCost', 'Annual\nEnergy', 'Annual\nMaintenance', 'Total PW']
            motor_a_values = [
                results['Motor A']['initial_cost'],
                results['Motor A']['annual_energy_cost'],
                results['Motor A']['annual_maintenance'],
                results['Motor A']['present_worth']
            ]
            motor_b_values = [
                results['Motor B']['initial_cost'],
                results['Motor B']['annual_energy_cost'],
                results['Motor B']['annual_maintenance'],
                results['Motor B']['present_worth']
            ]

            x = np.arange(len(categories))
            width = 0.35

            self.ax_cost_comparison.bar(x - width/2, motor_a_values, width, label='Motor A', color='steelblue')
            self.ax_cost_comparison.bar(x + width/2, motor_b_values, width, label='Motor B', color='coral')
            self.ax_cost_comparison.set_ylabel('Cost (Rs)', fontsize=9)
            self.ax_cost_comparison.set_title('Cost Comparison', fontsize=10, fontweight='bold')
            self.ax_cost_comparison.set_xticks(x)
            self.ax_cost_comparison.set_xticklabels(categories, fontsize=8)
            self.ax_cost_comparison.legend()
            self.ax_cost_comparison.grid(True, alpha=0.3)

            # Plot 2: Energy Consumption
            energy_categories = ['Full Load\nEnergy', 'Half Load\nEnergy', 'Total\nEnergy']

            # Calculate energy breakdown for Motor A
            full_load_hours_a = self.economic_analysis.operating_hours * self.economic_analysis.full_load_time
            half_load_hours_a = self.economic_analysis.operating_hours * self.economic_analysis.half_load_time
            energy_full_a = (self.economic_analysis.motor_power / self.economic_analysis.motor_a_eta_full) * full_load_hours_a
            energy_half_a = (self.economic_analysis.motor_power * 0.5 / self.economic_analysis.motor_a_eta_half) * half_load_hours_a

            # Calculate energy breakdown for Motor B
            energy_full_b = (self.economic_analysis.motor_power / self.economic_analysis.motor_b_eta_full) * full_load_hours_a
            energy_half_b = (self.economic_analysis.motor_power * 0.5 / self.economic_analysis.motor_b_eta_half) * half_load_hours_a

            motor_a_energy = [energy_full_a, energy_half_a, energy_full_a + energy_half_a]
            motor_b_energy = [energy_full_b, energy_half_b, energy_full_b + energy_half_b]

            x2 = np.arange(len(energy_categories))
            self.ax_energy_comparison.bar(x2 - width/2, motor_a_energy, width, label='Motor A', color='steelblue')
            self.ax_energy_comparison.bar(x2 + width/2, motor_b_energy, width, label='Motor B', color='coral')
            self.ax_energy_comparison.set_ylabel('Energy (kWh/year)', fontsize=9)
            self.ax_energy_comparison.set_title('Annual Energy Consumption', fontsize=10, fontweight='bold')
            self.ax_energy_comparison.set_xticks(x2)
            self.ax_energy_comparison.set_xticklabels(energy_categories, fontsize=8)
            self.ax_energy_comparison.legend()
            self.ax_energy_comparison.grid(True, alpha=0.3)

            # Plot 3: Cumulative Cost Over Time
            years = np.arange(0, self.economic_analysis.life_years + 1)
            cumulative_a = [results['Motor A']['initial_cost']]
            cumulative_b = [results['Motor B']['initial_cost']]

            for year in range(1, self.economic_analysis.life_years + 1):
                cumulative_a.append(cumulative_a[-1] + results['Motor A']['total_annual_cost'])
                cumulative_b.append(cumulative_b[-1] + results['Motor B']['total_annual_cost'])

            # Subtract salvage value at the end
            cumulative_a[-1] -= results['Motor A']['salvage_value']
            cumulative_b[-1] -= results['Motor B']['salvage_value']

            self.ax_cumulative.plot(years, cumulative_a, 'o-', label='Motor A', color='steelblue', linewidth=2, markersize=4)
            self.ax_cumulative.plot(years, cumulative_b, 's-', label='Motor B', color='coral', linewidth=2, markersize=4)
            self.ax_cumulative.set_xlabel('Years', fontsize=9)
            self.ax_cumulative.set_ylabel('Cumulative Cost (Rs)', fontsize=9)
            self.ax_cumulative.set_title('Cumulative Cost Over Lifetime', fontsize=10, fontweight='bold')
            self.ax_cumulative.legend()
            self.ax_cumulative.grid(True, alpha=0.3)

            # Plot 4: Cost Breakdown Pie Charts
            self.ax_breakdown.axis('equal')

            # Calculate breakdown for recommended motor
            recommended = results['recommendation']
            if recommended == 'Motor A':
                breakdown_data = results['Motor A']
                color_scheme = ['steelblue', 'lightblue', 'navy', 'skyblue']
            else:
                breakdown_data = results['Motor B']
                color_scheme = ['coral', 'lightsalmon', 'orangered', 'peachpuff']

            labels = ['Initial Cost', 'Energy (20y)', 'Maintenance (20y)', 'Salvage Value']
            sizes = [
                breakdown_data['initial_cost'],
                breakdown_data['annual_energy_cost'] * self.economic_analysis.life_years,
                breakdown_data['annual_maintenance'] * self.economic_analysis.life_years,
                -breakdown_data['salvage_value']  # Negative because it reduces cost
            ]

            # Filter out negative values for pie chart
            positive_sizes = [max(s, 0) for s in sizes]

            wedges, texts, autotexts = self.ax_breakdown.pie(positive_sizes, labels=labels, autopct='%1.1f%%',
                                                             colors=color_scheme, startangle=90)
            self.ax_breakdown.set_title(f'Cost Breakdown - {recommended}\n(Recommended)',
                                       fontsize=10, fontweight='bold')

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(8)
                autotext.set_fontweight('bold')

            for text in texts:
                text.set_fontsize(8)

            self.comp_fig.tight_layout()
            self.comp_fig.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Error generating comparison: {str(e)}")

    def on_window_resize(self, event):
        """Handle window resize event for auto-scaling"""
        # Only resize when window size actually changes
        if event.widget == self.root:
            try:
                # Update figure sizes based on window size
                new_width = max(event.width - 100, 800) / 100
                new_height = max(event.height - 100, 600) / 100

                # Update simulation figure
                if hasattr(self, 'sim_fig'):
                    self.sim_fig.set_size_inches(new_width * 0.6, new_height * 0.8)
                    self.sim_canvas.draw_idle()

                # Update comparison figure
                if hasattr(self, 'comp_fig'):
                    self.comp_fig.set_size_inches(new_width * 0.8, new_height * 0.8)
            except:
                pass  # Ignore resize errors


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = MotorAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
