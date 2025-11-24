"""
Comprehensive Electrical Engineering Simulator
Includes: Tariff Calculator, Power System Dynamics, ODE Solvers
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time


class ODESolver:
    """Base class for ODE solvers"""

    @staticmethod
    def euler(func, y0, t_span, dt):
        """Euler method for solving ODEs"""
        t0, tf = t_span
        t = np.arange(t0, tf, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(len(t) - 1):
            y[i + 1] = y[i] + dt * func(t[i], y[i])

        return t, y

    @staticmethod
    def rk45(func, y0, t_span, dt):
        """Runge-Kutta 4th/5th order method (RK45)"""
        t0, tf = t_span
        t = np.arange(t0, tf, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(len(t) - 1):
            h = dt
            k1 = func(t[i], y[i])
            k2 = func(t[i] + h/2, y[i] + h*k1/2)
            k3 = func(t[i] + h/2, y[i] + h*k2/2)
            k4 = func(t[i] + h, y[i] + h*k3)

            y[i + 1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

        return t, y

    @staticmethod
    def rk4_step(func, t, y, dt):
        """Single RK4 step for real-time integration"""
        k1 = func(t, y)
        k2 = func(t + dt / 2, y + dt * k1 / 2)
        k3 = func(t + dt / 2, y + dt * k2 / 2)
        k4 = func(t + dt, y + dt * k3)
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def euler_step(func, t, y, dt):
        """Single Euler step for real-time integration"""
        return y + dt * func(t, y)


class TariffCalculator:
    """Calculate electricity tariff and related parameters"""

    def __init__(self, energy_generated, max_demand, costs, running_percentages, loss_percentage):
        """
        Initialize tariff calculator

        Args:
            energy_generated: Annual energy in kWh
            max_demand: Maximum demand in MW
            costs: Dict with keys: fuel, generation, transmission, distribution (in lakhs)
            running_percentages: Dict with percentages for running charges
            loss_percentage: Transmission and distribution losses (%)
        """
        self.energy_generated = energy_generated
        self.max_demand = max_demand
        self.costs = costs
        self.running_percentages = running_percentages
        self.loss_percentage = loss_percentage

        self.results = {}
        self.calculate()

    def calculate(self):
        """Perform all tariff calculations"""
        # Total cost
        total_cost = sum(self.costs.values())

        # Running charges
        running_charges = {
            'fuel': self.costs['fuel'] * self.running_percentages['fuel'] / 100,
            'generation': self.costs['generation'] * self.running_percentages['generation'] / 100,
            'transmission': self.costs['transmission'] * self.running_percentages['transmission'] / 100,
            'distribution': self.costs['distribution'] * self.running_percentages['distribution'] / 100
        }
        total_running_charges = sum(running_charges.values())

        # Fixed charges
        fixed_charges = {
            'fuel': self.costs['fuel'] * (100 - self.running_percentages['fuel']) / 100,
            'generation': self.costs['generation'] * (100 - self.running_percentages['generation']) / 100,
            'transmission': self.costs['transmission'] * (100 - self.running_percentages['transmission']) / 100,
            'distribution': self.costs['distribution'] * (100 - self.running_percentages['distribution']) / 100
        }
        total_fixed_charges = sum(fixed_charges.values())

        # Energy calculations
        energy_loss = self.energy_generated * self.loss_percentage / 100
        energy_to_consumers = self.energy_generated - energy_loss

        # Load factor
        hours_per_year = 365 * 24
        average_demand_kw = energy_to_consumers / hours_per_year
        average_demand_mw = average_demand_kw / 1000
        load_factor = average_demand_mw / self.max_demand

        # Tariff components
        fixed_charge_per_kw = (total_fixed_charges * 1e5) / (self.max_demand * 1000)  # Rs/kW/year
        running_charge_per_kwh = (total_running_charges * 1e5) / energy_to_consumers  # Rs/kWh

        # Overall cost
        overall_cost_per_kwh = (total_cost * 1e5) / energy_to_consumers

        # Store results
        self.results = {
            'total_cost': total_cost,
            'total_running_charges': total_running_charges,
            'total_fixed_charges': total_fixed_charges,
            'running_charges': running_charges,
            'fixed_charges': fixed_charges,
            'energy_generated': self.energy_generated,
            'energy_loss': energy_loss,
            'energy_to_consumers': energy_to_consumers,
            'average_demand_mw': average_demand_mw,
            'load_factor': load_factor * 100,  # in percentage
            'fixed_charge_per_kw_year': fixed_charge_per_kw,
            'running_charge_per_kwh': running_charge_per_kwh,
            'overall_cost_per_kwh': overall_cost_per_kwh
        }

        return self.results


class PowerSystemDynamics:
    """Model power system dynamics using differential equations"""

    def __init__(self):
        self.H = 5.0  # Inertia constant (MJ/MVA)
        self.D = 1.0  # Damping coefficient
        self.Pm = 1.0  # Mechanical power (pu)
        self.Pe_max = 2.0  # Maximum electrical power (pu)
        self.omega_s = 2 * np.pi * 50  # Synchronous speed (rad/s)

    def swing_equation(self, t, state):
        """
        Swing equation for generator dynamics
        state = [delta, omega]
        delta: rotor angle (rad)
        omega: rotor speed deviation (rad/s)
        """
        delta, omega = state

        # Electrical power (simplified)
        Pe = self.Pe_max * np.sin(delta)

        # Swing equation: d(omega)/dt = (omega_s / (2*H)) * (Pm - Pe - D*omega)
        d_delta = omega
        d_omega = (self.omega_s / (2 * self.H)) * (self.Pm - Pe - self.D * omega)

        return np.array([d_delta, d_omega])

    def rlc_circuit(self, t, state, R, L, C, V_source):
        """
        RLC circuit differential equations
        state = [i, v_c]
        i: current through inductor
        v_c: voltage across capacitor
        """
        i, v_c = state

        # di/dt = (V_source - v_c - R*i) / L
        # dv_c/dt = i / C

        di_dt = (V_source * np.sin(2 * np.pi * 50 * t) - v_c - R * i) / L
        dv_c_dt = i / C

        return np.array([di_dt, dv_c_dt])


class TransformerEfficiencyCalculator:
    """Compute all-day efficiency for transformer operating schedules"""

    def __init__(self, rating_kva, core_loss_w, copper_loss_w, power_factor=1.0):
        self.rating_kva = rating_kva
        self.core_loss_w = core_loss_w
        self.copper_loss_w = copper_loss_w
        self.power_factor = power_factor

    def all_day_efficiency(self, schedule):
        """
        Calculate all-day efficiency.

        Args:
            schedule: Iterable of tuples (hours, load_fraction)
        """
        output_energy = 0.0
        copper_energy_loss = 0.0
        core_energy_loss = self.core_loss_w * 24 / 1000  # kWh over a day

        for hours, fraction in schedule:
            load_kw = self.rating_kva * fraction * self.power_factor
            output_energy += load_kw * hours
            copper_energy_loss += (fraction ** 2) * self.copper_loss_w * hours / 1000

        total_losses = core_energy_loss + copper_energy_loss
        efficiency = output_energy / (output_energy + total_losses) if (output_energy + total_losses) > 0 else 0

        return {
            'output_energy_kwh': output_energy,
            'core_loss_kwh': core_energy_loss,
            'copper_loss_kwh': copper_energy_loss,
            'total_losses_kwh': total_losses,
            'efficiency': efficiency * 100
        }


class ElectricalEngineeringSimulator(tk.Tk):
    """Main application class"""

    def __init__(self):
        super().__init__()

        self.title("Advanced Electrical Engineering Simulator")
        self.geometry("1400x900")
        self.minsize(1000, 700)

        # Simulation control
        self.running = False
        self.solver_type = "RK45"
        self.autoscale_enabled = True

        # Create GUI
        self.create_menus()
        self.create_widgets()

        # Bind resize event
        self.bind('<Configure>', self.on_resize)

    def create_menus(self):
        """Create a simple application menu"""
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Reset All", command=self.reset_all_views)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        self.autoscale_var = tk.BooleanVar(value=self.autoscale_enabled)
        view_menu.add_checkbutton(
            label="Autoscale Plots", onvalue=True, offvalue=False,
            variable=self.autoscale_var,
            command=self.toggle_autoscale
        )
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About",
                "Advanced Electrical Engineering Simulator\n"
                "Dynamic labs for tariff, transformers, and machine models"
            )
        )
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def create_widgets(self):
        """Create all GUI widgets"""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.create_tariff_tab()
        self.create_dynamics_tab()
        self.create_rlc_tab()
        self.create_transformer_tab()

    def create_tariff_tab(self):
        """Create tariff calculator tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Tariff Calculator")

        # Main container with two panels
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Inputs
        left_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding=10)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Right panel - Results
        right_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # Input fields
        row = 0

        # Energy generated
        ttk.Label(left_frame, text="Energy Generated (kWh/year):").grid(row=row, column=0, sticky='w', pady=5)
        self.energy_var = tk.StringVar(value="390000000")  # 39 × 10^7
        ttk.Entry(left_frame, textvariable=self.energy_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        # Maximum demand
        ttk.Label(left_frame, text="Maximum Demand (MW):").grid(row=row, column=0, sticky='w', pady=5)
        self.max_demand_var = tk.StringVar(value="130")
        ttk.Entry(left_frame, textvariable=self.max_demand_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        # Costs section
        ttk.Label(left_frame, text="\nCosts (in Lakhs Rs)", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1

        # Fuel cost
        ttk.Label(left_frame, text="Fuel:").grid(row=row, column=0, sticky='w', pady=5)
        self.fuel_cost_var = tk.StringVar(value="37.5")
        ttk.Entry(left_frame, textvariable=self.fuel_cost_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        # Generation cost
        ttk.Label(left_frame, text="Generation:").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_cost_var = tk.StringVar(value="18")
        ttk.Entry(left_frame, textvariable=self.gen_cost_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        # Transmission cost
        ttk.Label(left_frame, text="Transmission:").grid(row=row, column=0, sticky='w', pady=5)
        self.trans_cost_var = tk.StringVar(value="37.5")
        ttk.Entry(left_frame, textvariable=self.trans_cost_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        # Distribution cost
        ttk.Label(left_frame, text="Distribution:").grid(row=row, column=0, sticky='w', pady=5)
        self.dist_cost_var = tk.StringVar(value="25.5")
        ttk.Entry(left_frame, textvariable=self.dist_cost_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        # Running charges percentages
        ttk.Label(left_frame, text="\nRunning Charges (%)", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1

        ttk.Label(left_frame, text="Fuel:").grid(row=row, column=0, sticky='w', pady=5)
        self.fuel_run_var = tk.StringVar(value="90")
        ttk.Entry(left_frame, textvariable=self.fuel_run_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Generation:").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_run_var = tk.StringVar(value="10")
        ttk.Entry(left_frame, textvariable=self.gen_run_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Transmission:").grid(row=row, column=0, sticky='w', pady=5)
        self.trans_run_var = tk.StringVar(value="5")
        ttk.Entry(left_frame, textvariable=self.trans_run_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Distribution:").grid(row=row, column=0, sticky='w', pady=5)
        self.dist_run_var = tk.StringVar(value="7")
        ttk.Entry(left_frame, textvariable=self.dist_run_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        # Loss percentage
        ttk.Label(left_frame, text="\nSystem Losses (%):").grid(row=row, column=0, sticky='w', pady=5)
        self.loss_var = tk.StringVar(value="10")
        ttk.Entry(left_frame, textvariable=self.loss_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1

        # Power factor improvement study
        ttk.Label(left_frame, text="Power Factor Improvement", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=5
        )
        row += 1

        ttk.Label(left_frame, text="Annual Energy (kWh):").grid(row=row, column=0, sticky='w', pady=5)
        self.pf_energy_var = tk.StringVar(value="1000000")
        ttk.Entry(left_frame, textvariable=self.pf_energy_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Max Demand (kVA):").grid(row=row, column=0, sticky='w', pady=5)
        self.pf_demand_var = tk.StringVar(value="500")
        ttk.Entry(left_frame, textvariable=self.pf_demand_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Initial PF (lag):").grid(row=row, column=0, sticky='w', pady=5)
        self.pf_initial_var = tk.StringVar(value="0.707")
        ttk.Entry(left_frame, textvariable=self.pf_initial_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Target PF (lag):").grid(row=row, column=0, sticky='w', pady=5)
        self.pf_target_var = tk.StringVar(value="0.9")
        ttk.Entry(left_frame, textvariable=self.pf_target_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Demand Tariff (Rs/kVA·yr):").grid(row=row, column=0, sticky='w', pady=5)
        self.pf_tariff_var = tk.StringVar(value="75")
        ttk.Entry(left_frame, textvariable=self.pf_tariff_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Energy Tariff (paise/unit):").grid(row=row, column=0, sticky='w', pady=5)
        self.pf_energy_tariff_var = tk.StringVar(value="3")
        ttk.Entry(left_frame, textvariable=self.pf_energy_tariff_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Compensator Cost (Rs/kVA):").grid(row=row, column=0, sticky='w', pady=5)
        self.compensator_cost_var = tk.StringVar(value="45")
        ttk.Entry(left_frame, textvariable=self.compensator_cost_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Label(left_frame, text="Annual Rate on Plant (%):").grid(row=row, column=0, sticky='w', pady=5)
        self.plant_rate_var = tk.StringVar(value="10")
        ttk.Entry(left_frame, textvariable=self.plant_rate_var, width=20).grid(row=row, column=1, pady=5)
        row += 1

        ttk.Button(left_frame, text="PF Improvement Analysis", command=self.calculate_pf_improvement,
                   style='Accent.TButton').grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        # Calculate button
        ttk.Button(left_frame, text="Calculate Tariff", command=self.calculate_tariff,
                   style='Accent.TButton').grid(row=row, column=0, columnspan=2, pady=20)

        # Results text widget
        self.results_text = tk.Text(right_frame, wrap=tk.WORD, font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(right_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

    def create_dynamics_tab(self):
        """Create power system dynamics tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Generator Dynamics")

        # Control panel
        control_frame = ttk.LabelFrame(tab, text="Control Panel", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Parameters frame
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inertia constant
        ttk.Label(params_frame, text="Inertia Constant H (MJ/MVA):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.H_var = tk.DoubleVar(value=5.0)
        self.H_scale = ttk.Scale(params_frame, from_=1.0, to=10.0, variable=self.H_var,
                                  orient=tk.HORIZONTAL, length=200)
        self.H_scale.grid(row=0, column=1, padx=5, pady=5)
        self.H_label = ttk.Label(params_frame, text="5.0")
        self.H_label.grid(row=0, column=2, padx=5, pady=5)
        self.H_var.trace('w', lambda *args: self.H_label.config(text=f"{self.H_var.get():.2f}"))

        # Damping coefficient
        ttk.Label(params_frame, text="Damping Coefficient D:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.D_var = tk.DoubleVar(value=1.0)
        self.D_scale = ttk.Scale(params_frame, from_=0.1, to=5.0, variable=self.D_var,
                                  orient=tk.HORIZONTAL, length=200)
        self.D_scale.grid(row=1, column=1, padx=5, pady=5)
        self.D_label = ttk.Label(params_frame, text="1.0")
        self.D_label.grid(row=1, column=2, padx=5, pady=5)
        self.D_var.trace('w', lambda *args: self.D_label.config(text=f"{self.D_var.get():.2f}"))

        # Mechanical power
        ttk.Label(params_frame, text="Mechanical Power Pm (pu):").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.Pm_var = tk.DoubleVar(value=1.0)
        self.Pm_scale = ttk.Scale(params_frame, from_=0.1, to=2.0, variable=self.Pm_var,
                                   orient=tk.HORIZONTAL, length=200)
        self.Pm_scale.grid(row=2, column=1, padx=5, pady=5)
        self.Pm_label = ttk.Label(params_frame, text="1.0")
        self.Pm_label.grid(row=2, column=2, padx=5, pady=5)
        self.Pm_var.trace('w', lambda *args: self.Pm_label.config(text=f"{self.Pm_var.get():.2f}"))

        # Initial rotor angle
        ttk.Label(params_frame, text="Initial Rotor Angle δ₀ (deg):").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.delta0_var = tk.DoubleVar(value=30.0)
        self.delta0_scale = ttk.Scale(params_frame, from_=0.0, to=90.0, variable=self.delta0_var,
                                       orient=tk.HORIZONTAL, length=200)
        self.delta0_scale.grid(row=3, column=1, padx=5, pady=5)
        self.delta0_label = ttk.Label(params_frame, text="30.0")
        self.delta0_label.grid(row=3, column=2, padx=5, pady=5)
        self.delta0_var.trace('w', lambda *args: self.delta0_label.config(text=f"{self.delta0_var.get():.1f}"))

        # Simulation duration
        ttk.Label(params_frame, text="Simulation Time (s):").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.time_var = tk.DoubleVar(value=10.0)
        self.time_scale = ttk.Scale(params_frame, from_=1.0, to=30.0, variable=self.time_var,
                                     orient=tk.HORIZONTAL, length=200)
        self.time_scale.grid(row=4, column=1, padx=5, pady=5)
        self.time_label = ttk.Label(params_frame, text="10.0")
        self.time_label.grid(row=4, column=2, padx=5, pady=5)
        self.time_var.trace('w', lambda *args: self.time_label.config(text=f"{self.time_var.get():.1f}"))

        # Time-step
        ttk.Label(params_frame, text="Time Step Δt (s):").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.dt_var = tk.DoubleVar(value=0.01)
        self.dt_scale = ttk.Scale(params_frame, from_=0.001, to=0.05, variable=self.dt_var,
                                   orient=tk.HORIZONTAL, length=200)
        self.dt_scale.grid(row=5, column=1, padx=5, pady=5)
        self.dt_label = ttk.Label(params_frame, text="0.01")
        self.dt_label.grid(row=5, column=2, padx=5, pady=5)
        self.dt_var.trace('w', lambda *args: self.dt_label.config(text=f"{self.dt_var.get():.3f}"))

        # Solver selection
        ttk.Label(params_frame, text="ODE Solver:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.solver_var = tk.StringVar(value="RK45")
        solver_combo = ttk.Combobox(params_frame, textvariable=self.solver_var,
                                     values=["RK45", "Euler"], state='readonly', width=18)
        solver_combo.grid(row=6, column=1, padx=5, pady=5, sticky='w')

        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(side=tk.RIGHT, padx=10)

        self.start_btn = ttk.Button(buttons_frame, text="Start", command=self.start_simulation,
                                     style='Accent.TButton', width=15)
        self.start_btn.pack(pady=5)

        self.stop_btn = ttk.Button(buttons_frame, text="Stop", command=self.stop_simulation,
                                    state='disabled', width=15)
        self.stop_btn.pack(pady=5)

        self.reset_btn = ttk.Button(buttons_frame, text="Reset", command=self.reset_simulation, width=15)
        self.reset_btn.pack(pady=5)

        # Visualization frame
        viz_frame = ttk.LabelFrame(tab, text="Real-Time Visualization", padding=10)
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create matplotlib figure
        self.fig_dynamics = Figure(figsize=(12, 6))
        self.ax1 = self.fig_dynamics.add_subplot(2, 1, 1)
        self.ax2 = self.fig_dynamics.add_subplot(2, 1, 2)

        self.canvas_dynamics = FigureCanvasTkAgg(self.fig_dynamics, master=viz_frame)
        self.canvas_dynamics.draw()
        self.canvas_dynamics.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize plots
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Rotor Angle δ (deg)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Generator Rotor Angle (Swing Equation)')

        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Speed Deviation Δω (rad/s)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_title('Rotor Speed Deviation')

        self.fig_dynamics.tight_layout()

    def create_rlc_tab(self):
        """Create RLC circuit dynamics tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="RLC Circuit Dynamics")

        # Control panel
        control_frame = ttk.LabelFrame(tab, text="Circuit Parameters", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Parameters frame
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Resistance
        ttk.Label(params_frame, text="Resistance R (Ω):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.R_var = tk.DoubleVar(value=10.0)
        self.R_scale = ttk.Scale(params_frame, from_=1.0, to=100.0, variable=self.R_var,
                                  orient=tk.HORIZONTAL, length=200)
        self.R_scale.grid(row=0, column=1, padx=5, pady=5)
        self.R_label = ttk.Label(params_frame, text="10.0")
        self.R_label.grid(row=0, column=2, padx=5, pady=5)
        self.R_var.trace('w', lambda *args: self.R_label.config(text=f"{self.R_var.get():.1f}"))

        # Inductance
        ttk.Label(params_frame, text="Inductance L (H):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.L_var = tk.DoubleVar(value=0.1)
        self.L_scale = ttk.Scale(params_frame, from_=0.01, to=1.0, variable=self.L_var,
                                  orient=tk.HORIZONTAL, length=200)
        self.L_scale.grid(row=1, column=1, padx=5, pady=5)
        self.L_label = ttk.Label(params_frame, text="0.1")
        self.L_label.grid(row=1, column=2, padx=5, pady=5)
        self.L_var.trace('w', lambda *args: self.L_label.config(text=f"{self.L_var.get():.3f}"))

        # Capacitance
        ttk.Label(params_frame, text="Capacitance C (F):").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.C_var = tk.DoubleVar(value=0.001)
        self.C_scale = ttk.Scale(params_frame, from_=0.0001, to=0.01, variable=self.C_var,
                                  orient=tk.HORIZONTAL, length=200)
        self.C_scale.grid(row=2, column=1, padx=5, pady=5)
        self.C_label = ttk.Label(params_frame, text="0.001")
        self.C_label.grid(row=2, column=2, padx=5, pady=5)
        self.C_var.trace('w', lambda *args: self.C_label.config(text=f"{self.C_var.get():.4f}"))

        # Voltage source
        ttk.Label(params_frame, text="Source Voltage V (V):").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.V_var = tk.DoubleVar(value=230.0)
        self.V_scale = ttk.Scale(params_frame, from_=50.0, to=500.0, variable=self.V_var,
                                  orient=tk.HORIZONTAL, length=200)
        self.V_scale.grid(row=3, column=1, padx=5, pady=5)
        self.V_label = ttk.Label(params_frame, text="230.0")
        self.V_label.grid(row=3, column=2, padx=5, pady=5)
        self.V_var.trace('w', lambda *args: self.V_label.config(text=f"{self.V_var.get():.1f}"))

        # Solver selection
        ttk.Label(params_frame, text="ODE Solver:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.rlc_solver_var = tk.StringVar(value="RK45")
        solver_combo = ttk.Combobox(params_frame, textvariable=self.rlc_solver_var,
                                     values=["RK45", "Euler"], state='readonly', width=18)
        solver_combo.grid(row=4, column=1, padx=5, pady=5, sticky='w')

        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(side=tk.RIGHT, padx=10)

        self.rlc_start_btn = ttk.Button(buttons_frame, text="Start", command=self.start_rlc_simulation,
                                        style='Accent.TButton', width=15)
        self.rlc_start_btn.pack(pady=5)

        self.rlc_stop_btn = ttk.Button(buttons_frame, text="Stop", command=self.stop_rlc_simulation,
                                       state='disabled', width=15)
        self.rlc_stop_btn.pack(pady=5)

        self.rlc_reset_btn = ttk.Button(buttons_frame, text="Reset", command=self.reset_rlc_simulation, width=15)
        self.rlc_reset_btn.pack(pady=5)

        # Visualization frame
        viz_frame = ttk.LabelFrame(tab, text="Real-Time Visualization", padding=10)
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create matplotlib figure
        self.fig_rlc = Figure(figsize=(12, 6))
        self.ax_rlc1 = self.fig_rlc.add_subplot(2, 1, 1)
        self.ax_rlc2 = self.fig_rlc.add_subplot(2, 1, 2)

        self.canvas_rlc = FigureCanvasTkAgg(self.fig_rlc, master=viz_frame)
        self.canvas_rlc.draw()
        self.canvas_rlc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize plots
        self.ax_rlc1.set_xlabel('Time (s)')
        self.ax_rlc1.set_ylabel('Current i (A)')
        self.ax_rlc1.grid(True, alpha=0.3)
        self.ax_rlc1.set_title('RLC Circuit - Current')

        self.ax_rlc2.set_xlabel('Time (s)')
        self.ax_rlc2.set_ylabel('Capacitor Voltage Vc (V)')
        self.ax_rlc2.grid(True, alpha=0.3)
        self.ax_rlc2.set_title('RLC Circuit - Capacitor Voltage')

        self.fig_rlc.tight_layout()

    def calculate_tariff(self):
        """Calculate and display tariff"""
        try:
            # Get input values
            energy_generated = float(self.energy_var.get())
            max_demand = float(self.max_demand_var.get())

            costs = {
                'fuel': float(self.fuel_cost_var.get()),
                'generation': float(self.gen_cost_var.get()),
                'transmission': float(self.trans_cost_var.get()),
                'distribution': float(self.dist_cost_var.get())
            }

            running_percentages = {
                'fuel': float(self.fuel_run_var.get()),
                'generation': float(self.gen_run_var.get()),
                'transmission': float(self.trans_run_var.get()),
                'distribution': float(self.dist_run_var.get())
            }

            loss_percentage = float(self.loss_var.get())

            # Calculate tariff
            calculator = TariffCalculator(energy_generated, max_demand, costs,
                                         running_percentages, loss_percentage)
            results = calculator.results

            # Display results
            self.results_text.delete(1.0, tk.END)

            output = "=" * 70 + "\n"
            output += "TARIFF CALCULATION RESULTS\n"
            output += "=" * 70 + "\n\n"

            output += "COST BREAKDOWN:\n"
            output += "-" * 70 + "\n"
            output += f"Total Cost:                    {results['total_cost']:>15.2f} lakhs\n"
            output += f"Total Running Charges:         {results['total_running_charges']:>15.2f} lakhs\n"
            output += f"Total Fixed Charges:           {results['total_fixed_charges']:>15.2f} lakhs\n\n"

            output += "RUNNING CHARGES BREAKDOWN:\n"
            output += "-" * 70 + "\n"
            for key, value in results['running_charges'].items():
                output += f"{key.capitalize():20s}       {value:>15.2f} lakhs\n"

            output += "\nFIXED CHARGES BREAKDOWN:\n"
            output += "-" * 70 + "\n"
            for key, value in results['fixed_charges'].items():
                output += f"{key.capitalize():20s}       {value:>15.2f} lakhs\n"

            output += "\n" + "=" * 70 + "\n"
            output += "ENERGY AND DEMAND ANALYSIS:\n"
            output += "=" * 70 + "\n"
            output += f"Energy Generated:              {results['energy_generated']:>15.0f} kWh\n"
            output += f"Energy Loss:                   {results['energy_loss']:>15.0f} kWh\n"
            output += f"Energy to Consumers:           {results['energy_to_consumers']:>15.0f} kWh\n"
            output += f"Maximum Demand:                {max_demand:>15.2f} MW\n"
            output += f"Average Demand:                {results['average_demand_mw']:>15.2f} MW\n"
            output += f"Load Factor:                   {results['load_factor']:>15.2f} %\n\n"

            output += "=" * 70 + "\n"
            output += "TARIFF STRUCTURE:\n"
            output += "=" * 70 + "\n"
            output += f"Fixed Charge:                  {results['fixed_charge_per_kw_year']:>15.2f} Rs/kW/year\n"
            output += f"Running Charge:                {results['running_charge_per_kwh']:>15.4f} Rs/kWh\n"
            output += f"Overall Cost per Unit:         {results['overall_cost_per_kwh']:>15.4f} Rs/kWh\n"
            output += f"Overall Cost per Unit:         {results['overall_cost_per_kwh']*100:>15.4f} paise/kWh\n\n"

            output += "=" * 70 + "\n"
            output += "TARIFF FORMULA:\n"
            output += "=" * 70 + "\n"
            output += f"Annual Bill = Fixed Charge × Maximum Demand + Running Charge × Energy Consumed\n"
            output += f"            = {results['fixed_charge_per_kw_year']:.2f} × (kW) + "
            output += f"{results['running_charge_per_kwh']:.4f} × (kWh)\n\n"

            self.results_text.insert(1.0, output)

        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def calculate_pf_improvement(self):
        """Calculate annual cost and savings for power factor correction"""
        try:
            energy_kwh = float(self.pf_energy_var.get())
            max_demand_kva = float(self.pf_demand_var.get())
            pf_initial = float(self.pf_initial_var.get())
            pf_target = float(self.pf_target_var.get())
            demand_tariff = float(self.pf_tariff_var.get())
            energy_tariff_paise = float(self.pf_energy_tariff_var.get())
            compensator_cost = float(self.compensator_cost_var.get())
            annual_rate = float(self.plant_rate_var.get())

            # Base values
            active_power_kw = max_demand_kva * pf_initial
            kvar_initial = active_power_kw * np.tan(np.arccos(pf_initial))
            kvar_target = active_power_kw * np.tan(np.arccos(pf_target))
            kvar_required = max(kvar_initial - kvar_target, 0)

            new_demand_kva = active_power_kw / pf_target

            # Tariff calculations
            energy_cost_rs = energy_kwh * (energy_tariff_paise / 100)
            annual_cost_before = max_demand_kva * demand_tariff + energy_cost_rs
            annual_cost_after = new_demand_kva * demand_tariff + energy_cost_rs

            # Plant economics
            plant_capex = compensator_cost * kvar_required
            annual_plant_cost = annual_rate / 100 * plant_capex

            gross_saving = annual_cost_before - annual_cost_after
            net_saving = gross_saving - annual_plant_cost

            pf_output = "\n" + "=" * 70 + "\n"
            pf_output += "POWER FACTOR CORRECTION STUDY\n"
            pf_output += "=" * 70 + "\n"
            pf_output += f"Active Power (kW):             {active_power_kw:>12.2f}\n"
            pf_output += f"Initial Reactive Power (kVAr): {kvar_initial:>12.2f}\n"
            pf_output += f"Target Reactive Power (kVAr):  {kvar_target:>12.2f}\n"
            pf_output += f"Required Compensator (kVAr):   {kvar_required:>12.2f}\n"
            pf_output += f"New Maximum Demand (kVA):      {new_demand_kva:>12.2f}\n\n"

            pf_output += "ANNUAL COSTING (Rs)\n"
            pf_output += "-" * 70 + "\n"
            pf_output += f"Before Improvement:            {annual_cost_before:>12.2f}\n"
            pf_output += f"After Improvement:             {annual_cost_after:>12.2f}\n"
            pf_output += f"Gross Saving:                  {gross_saving:>12.2f}\n"
            pf_output += f"Annual Plant Cost (@{annual_rate:.1f}%):   {annual_plant_cost:>12.2f}\n"
            pf_output += f"Net Annual Saving:             {net_saving:>12.2f}\n"

            self.results_text.insert(tk.END, pf_output)
            self.results_text.see(tk.END)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for the PF study")
        except Exception as exc:
            messagebox.showerror("Error", f"An error occurred: {exc}")

    def start_simulation(self):
        """Start generator dynamics simulation"""
        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        # Run simulation in separate thread
        thread = threading.Thread(target=self.run_dynamics_simulation)
        thread.daemon = True
        thread.start()

    def stop_simulation(self):
        """Stop generator dynamics simulation"""
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def reset_simulation(self):
        """Reset generator dynamics simulation"""
        self.stop_simulation()
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Rotor Angle δ (deg)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Generator Rotor Angle (Swing Equation)')

        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Speed Deviation Δω (rad/s)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_title('Rotor Speed Deviation')

        self.canvas_dynamics.draw()

    def run_dynamics_simulation(self):
        """Run the dynamics simulation"""
        # Get parameters
        dynamics = PowerSystemDynamics()
        dynamics.H = self.H_var.get()
        dynamics.D = self.D_var.get()
        dynamics.Pm = self.Pm_var.get()

        # Initial conditions
        delta0_deg = self.delta0_var.get()
        delta0_rad = np.deg2rad(delta0_deg)
        omega0 = 0.0
        y0 = np.array([delta0_rad, omega0])

        total_time = self.time_var.get()
        dt = self.dt_var.get()
        solver = ODESolver()

        # Prepare plots for live update
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Rotor Angle δ (deg)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title(f'Generator Rotor Angle (Swing Equation) - Solver: {self.solver_var.get()}')

        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Speed Deviation Δω (rad/s)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_title('Rotor Speed Deviation')

        t_values = [0.0]
        states = [y0]
        redraw_interval = max(1, int(0.1 / dt))

        current_t = 0.0
        current_state = y0

        while self.running and current_t < total_time:
            if self.solver_var.get() == "RK45":
                next_state = solver.rk4_step(dynamics.swing_equation, current_t, current_state, dt)
            else:
                next_state = solver.euler_step(dynamics.swing_equation, current_t, current_state, dt)

            current_t += dt
            current_state = next_state
            t_values.append(current_t)
            states.append(current_state)

            if len(t_values) % redraw_interval == 0:
                y_array = np.array(states)
                delta_deg = np.rad2deg(y_array[:, 0])
                omega = y_array[:, 1]
                self.ax1.plot(t_values, delta_deg, 'b-', linewidth=2)
                self.ax2.plot(t_values, omega, 'r-', linewidth=2)
                self.fig_dynamics.tight_layout()
                self.canvas_dynamics.draw()
                time.sleep(0.01)

        # Final update
        y_array = np.array(states)
        delta_deg = np.rad2deg(y_array[:, 0])
        omega = y_array[:, 1]
        self.ax1.plot(t_values, delta_deg, 'b-', linewidth=2)
        self.ax2.plot(t_values, omega, 'r-', linewidth=2)
        self.fig_dynamics.tight_layout()
        self.canvas_dynamics.draw()

        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

    def start_rlc_simulation(self):
        """Start RLC circuit simulation"""
        self.rlc_running = True
        self.rlc_start_btn.config(state='disabled')
        self.rlc_stop_btn.config(state='normal')

        # Run simulation in separate thread
        thread = threading.Thread(target=self.run_rlc_simulation_calc)
        thread.daemon = True
        thread.start()

    def stop_rlc_simulation(self):
        """Stop RLC circuit simulation"""
        self.rlc_running = False
        self.rlc_start_btn.config(state='normal')
        self.rlc_stop_btn.config(state='disabled')

    def reset_rlc_simulation(self):
        """Reset RLC circuit simulation"""
        self.stop_rlc_simulation()
        self.ax_rlc1.clear()
        self.ax_rlc2.clear()

        self.ax_rlc1.set_xlabel('Time (s)')
        self.ax_rlc1.set_ylabel('Current i (A)')
        self.ax_rlc1.grid(True, alpha=0.3)
        self.ax_rlc1.set_title('RLC Circuit - Current')

        self.ax_rlc2.set_xlabel('Time (s)')
        self.ax_rlc2.set_ylabel('Capacitor Voltage Vc (V)')
        self.ax_rlc2.grid(True, alpha=0.3)
        self.ax_rlc2.set_title('RLC Circuit - Capacitor Voltage')

        self.canvas_rlc.draw()

    def run_rlc_simulation_calc(self):
        """Run the RLC circuit simulation"""
        # Get parameters
        R = self.R_var.get()
        L = self.L_var.get()
        C = self.C_var.get()
        V = self.V_var.get()

        dynamics = PowerSystemDynamics()

        # Initial conditions
        y0 = np.array([0.0, 0.0])  # [current, capacitor voltage]

        # Time parameters
        t_span = (0, 0.5)
        dt = 0.0001

        # Create RLC function with parameters
        rlc_func = lambda t, y: dynamics.rlc_circuit(t, y, R, L, C, V)

        # Solve ODE
        solver = ODESolver()
        if self.rlc_solver_var.get() == "RK45":
            t, y = solver.rk45(rlc_func, y0, t_span, dt)
        else:
            t, y = solver.euler(rlc_func, y0, t_span, dt)

        current = y[:, 0]
        voltage_c = y[:, 1]

        # Update plots
        self.ax_rlc1.clear()
        self.ax_rlc2.clear()

        self.ax_rlc1.plot(t, current, 'b-', linewidth=2)
        self.ax_rlc1.set_xlabel('Time (s)')
        self.ax_rlc1.set_ylabel('Current i (A)')
        self.ax_rlc1.grid(True, alpha=0.3)
        self.ax_rlc1.set_title(f'RLC Circuit - Current (Solver: {self.rlc_solver_var.get()})')

        self.ax_rlc2.plot(t, voltage_c, 'r-', linewidth=2)
        self.ax_rlc2.set_xlabel('Time (s)')
        self.ax_rlc2.set_ylabel('Capacitor Voltage Vc (V)')
        self.ax_rlc2.grid(True, alpha=0.3)
        self.ax_rlc2.set_title('RLC Circuit - Capacitor Voltage')

        self.fig_rlc.tight_layout()
        self.canvas_rlc.draw()

        self.rlc_running = False
        self.rlc_start_btn.config(state='normal')
        self.rlc_stop_btn.config(state='disabled')

    def on_resize(self, event):
        """Handle window resize event"""
        if event.widget is not self:
            return

        new_w = max(self.winfo_width() - 40, 600)
        new_h = max(self.winfo_height() - 120, 400)

        if hasattr(self, 'canvas_dynamics'):
            self.canvas_dynamics.get_tk_widget().config(width=new_w, height=new_h // 2)
        if hasattr(self, 'canvas_rlc'):
            self.canvas_rlc.get_tk_widget().config(width=new_w, height=new_h // 2)
        if hasattr(self, 'canvas_transformer'):
            self.canvas_transformer.get_tk_widget().config(width=new_w, height=new_h // 3)

        self.update_idletasks()

        if self.autoscale_enabled:
            for axis in [getattr(self, 'ax1', None), getattr(self, 'ax2', None),
                         getattr(self, 'ax_rlc1', None), getattr(self, 'ax_rlc2', None)]:
                if axis:
                    axis.relim()
                    axis.autoscale_view()
            if hasattr(self, 'fig_transformer'):
                for axis in self.fig_transformer.axes:
                    axis.relim()
                    axis.autoscale_view()
            if hasattr(self, 'canvas_dynamics'):
                self.canvas_dynamics.draw()
            if hasattr(self, 'canvas_rlc'):
                self.canvas_rlc.draw()
            if hasattr(self, 'canvas_transformer'):
                self.canvas_transformer.draw()

    def create_transformer_tab(self):
        """Create transformer all-day efficiency comparison tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Transformer Lab")

        top_frame = ttk.Frame(tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Lighting Load Schedule (hours):", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, sticky='w', pady=5
        )
        self.full_load_hours = tk.DoubleVar(value=4.0)
        self.half_load_hours = tk.DoubleVar(value=8.0)
        self.no_load_hours = tk.DoubleVar(value=12.0)

        ttk.Label(top_frame, text="Full-load hours:").grid(row=1, column=0, sticky='w', padx=5)
        ttk.Scale(top_frame, from_=0, to=24, variable=self.full_load_hours, orient=tk.HORIZONTAL, length=200,
                  command=lambda _v: self.sync_no_load_hours()).grid(row=1, column=1, padx=5)
        self.full_label = ttk.Label(top_frame, text="4.0")
        self.full_label.grid(row=1, column=2, padx=5)
        self.full_load_hours.trace('w', lambda *args: self.full_label.config(text=f"{self.full_load_hours.get():.1f}"))

        ttk.Label(top_frame, text="Half-load hours:").grid(row=2, column=0, sticky='w', padx=5)
        ttk.Scale(top_frame, from_=0, to=24, variable=self.half_load_hours, orient=tk.HORIZONTAL, length=200,
                  command=lambda _v: self.sync_no_load_hours()).grid(row=2, column=1, padx=5)
        self.half_label = ttk.Label(top_frame, text="8.0")
        self.half_label.grid(row=2, column=2, padx=5)
        self.half_load_hours.trace('w', lambda *args: self.half_label.config(text=f"{self.half_load_hours.get():.1f}"))

        ttk.Label(top_frame, text="No-load hours (auto):").grid(row=3, column=0, sticky='w', padx=5)
        self.no_label = ttk.Label(top_frame, text="12.0")
        self.no_label.grid(row=3, column=1, sticky='w', padx=5)

        ttk.Separator(top_frame, orient=tk.VERTICAL).grid(row=0, column=3, rowspan=4, sticky='ns', padx=10)

        params_frame = ttk.LabelFrame(tab, text="Transformer Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(params_frame, text="Rating (kVA):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.rating_var = tk.DoubleVar(value=40.0)
        ttk.Entry(params_frame, textvariable=self.rating_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Power Factor (lighting pf≈1.0):").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        self.pf_var = tk.DoubleVar(value=1.0)
        ttk.Entry(params_frame, textvariable=self.pf_var, width=10).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(params_frame, text="Core Loss A (W):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.core_a_var = tk.DoubleVar(value=500)
        ttk.Entry(params_frame, textvariable=self.core_a_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Copper Loss A (W):").grid(row=1, column=2, sticky='w', padx=5, pady=5)
        self.copper_a_var = tk.DoubleVar(value=500)
        ttk.Entry(params_frame, textvariable=self.copper_a_var, width=10).grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(params_frame, text="Core Loss B (W):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.core_b_var = tk.DoubleVar(value=250)
        ttk.Entry(params_frame, textvariable=self.core_b_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Copper Loss B (W):").grid(row=2, column=2, sticky='w', padx=5, pady=5)
        self.copper_b_var = tk.DoubleVar(value=750)
        ttk.Entry(params_frame, textvariable=self.copper_b_var, width=10).grid(row=2, column=3, padx=5, pady=5)

        ttk.Button(params_frame, text="Compare All-Day Efficiency", style='Accent.TButton',
                   command=self.calculate_transformer_efficiency).grid(row=3, column=0, columnspan=4, pady=10)

        results_frame = ttk.Frame(tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.transformer_text = tk.Text(results_frame, height=10, wrap=tk.WORD, font=('Courier', 10))
        self.transformer_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(results_frame, command=self.transformer_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.transformer_text.configure(yscrollcommand=scroll.set)

        # Visualization
        self.fig_transformer = Figure(figsize=(8, 4))
        self.ax_trans_load = self.fig_transformer.add_subplot(121)
        self.ax_trans_eff = self.fig_transformer.add_subplot(122)

        canvas_frame = ttk.Frame(tab)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_transformer = FigureCanvasTkAgg(self.fig_transformer, master=canvas_frame)
        self.canvas_transformer.draw()
        self.canvas_transformer.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_transformer_plot([], {})

    def toggle_autoscale(self):
        """Toggle autoscaling of plots"""
        self.autoscale_enabled = not self.autoscale_enabled
        if hasattr(self, 'autoscale_var'):
            self.autoscale_var.set(self.autoscale_enabled)

    def reset_all_views(self):
        """Reset plots and text outputs"""
        self.reset_simulation()
        self.reset_rlc_simulation()
        self.transformer_text.delete(1.0, tk.END)
        self.update_transformer_plot([], {})

    def sync_no_load_hours(self):
        """Keep the schedule totaling 24 hours"""
        remaining = max(24.0 - self.full_load_hours.get() - self.half_load_hours.get(), 0)
        self.no_load_hours.set(remaining)
        self.no_label.config(text=f"{remaining:.1f}")

    def calculate_transformer_efficiency(self):
        """Compute and display all-day efficiency for two transformers"""
        try:
            schedule = [
                (self.full_load_hours.get(), 1.0),
                (self.half_load_hours.get(), 0.5),
                (self.no_load_hours.get(), 0.0),
            ]

            rating = self.rating_var.get()
            pf = self.pf_var.get()

            tra = TransformerEfficiencyCalculator(rating, self.core_a_var.get(), self.copper_a_var.get(), pf)
            trb = TransformerEfficiencyCalculator(rating, self.core_b_var.get(), self.copper_b_var.get(), pf)

            res_a = tra.all_day_efficiency(schedule)
            res_b = trb.all_day_efficiency(schedule)

            summary = "Transformer All-Day Efficiency Study\n" + "=" * 70 + "\n"
            summary += f"Schedule: {schedule[0][0]:.1f}h @ 100%, {schedule[1][0]:.1f}h @ 50%, {schedule[2][0]:.1f}h @ 0%\n"
            summary += f"Rating: {rating:.1f} kVA, Power Factor: {pf:.2f}\n\n"

            for name, res in [("Transformer A", res_a), ("Transformer B", res_b)]:
                summary += f"{name}:\n"
                summary += f"  Output Energy: {res['output_energy_kwh']:.2f} kWh\n"
                summary += f"  Core Loss:     {res['core_loss_kwh']:.2f} kWh\n"
                summary += f"  Copper Loss:   {res['copper_loss_kwh']:.2f} kWh\n"
                summary += f"  Total Losses:  {res['total_losses_kwh']:.2f} kWh\n"
                summary += f"  All-Day Eff.:  {res['efficiency']:.2f}%\n\n"

            better = "A" if res_a['efficiency'] > res_b['efficiency'] else "B"
            summary += f"More efficient for the given lighting duty: Transformer {better}.\n"

            self.transformer_text.delete(1.0, tk.END)
            self.transformer_text.insert(1.0, summary)
            self.update_transformer_plot(schedule, {"A": res_a, "B": res_b})

        except Exception as exc:
            messagebox.showerror("Calculation Error", f"Unable to compute efficiency: {exc}")

    def update_transformer_plot(self, schedule, results):
        """Update transformer visualizations"""
        self.ax_trans_load.clear()
        self.ax_trans_eff.clear()

        if schedule:
            labels = ['Full Load', 'Half Load', 'No Load']
            hours = [schedule[0][0], schedule[1][0], schedule[2][0]]
            self.ax_trans_load.bar(labels, hours, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            self.ax_trans_load.set_ylabel('Hours per day')
            self.ax_trans_load.set_title('Daily Loading Profile')
            self.ax_trans_load.grid(True, axis='y', alpha=0.3)

        if results:
            names = list(results.keys())
            efficiencies = [results[n]['efficiency'] for n in names]
            self.ax_trans_eff.bar(names, efficiencies, color=['#9467bd', '#8c564b'])
            self.ax_trans_eff.set_ylim(0, 100)
            self.ax_trans_eff.set_ylabel('All-Day Efficiency (%)')
            self.ax_trans_eff.set_title('Transformer Comparison')
            self.ax_trans_eff.grid(True, axis='y', alpha=0.3)

        self.fig_transformer.tight_layout()
        self.canvas_transformer.draw()


def main():
    """Main entry point"""
    app = ElectricalEngineeringSimulator()
    app.mainloop()


if __name__ == "__main__":
    main()
