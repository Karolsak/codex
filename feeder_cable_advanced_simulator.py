"""
Advanced Electrical Engineering Simulator with Feeder Cable Optimization
Includes: Feeder Cable Calculator, Dynamic System Simulation, ODE Solvers,
          Real-time Visualization, and Multiple Electrical Engineering Tools
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math


class ODESolver:
    """Base class for ODE solvers"""

    @staticmethod
    def euler(f, y0, t):
        """Euler method for solving ODEs"""
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            y[i + 1] = y[i] + dt * f(t[i], y[i])
        return y

    @staticmethod
    def rk45(f, y0, t):
        """Runge-Kutta 4th/5th order method (RK45)"""
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            k1 = f(t[i], y[i])
            k2 = f(t[i] + dt/4, y[i] + dt*k1/4)
            k3 = f(t[i] + 3*dt/8, y[i] + dt*(3*k1 + 9*k2)/32)
            k4 = f(t[i] + 12*dt/13, y[i] + dt*(1932*k1 - 7200*k2 + 7296*k3)/2197)
            k5 = f(t[i] + dt, y[i] + dt*(439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
            k6 = f(t[i] + dt/2, y[i] + dt*(-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))

            # 4th order solution
            y[i + 1] = y[i] + dt * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
        return y


class FeederCableOptimizer:
    """Optimizes feeder cable cross-section for economic operation"""

    def __init__(self, voltage, length, max_current, months_operation,
                 resistance_per_km_per_sqcm, cable_cost_a, cable_cost_b,
                 interest_depreciation, energy_cost):
        self.V = voltage  # Voltage in V
        self.L = length  # Length in km
        self.I_max = max_current  # Maximum current in A
        self.months = months_operation  # Months of operation per year
        self.rho = resistance_per_km_per_sqcm  # Resistance coefficient
        self.cost_a = cable_cost_a  # Cable cost coefficient A
        self.cost_b = cable_cost_b  # Cable cost coefficient B
        self.i_d = interest_depreciation  # Interest & depreciation rate
        self.energy_cost = energy_cost  # Energy cost in Rs/kWh

    def calculate_optimal_cross_section(self):
        """Calculate the most economical cross-section"""
        # For 2-core cable
        # Annual operating hours
        hours = self.months * 30 * 24

        # Total resistance R = (rho * L * 2) / A
        resistance_factor = self.rho * self.L * 2

        # Annual energy loss = I²R * hours / 1000 = I² * resistance_factor * hours / (1000 * A)
        energy_loss_factor = (self.I_max ** 2) * resistance_factor * hours / 1000

        # Annual capital cost = (cost_a * A + cost_b) * L * 1000 * i_d
        capital_cost_factor = (self.cost_a) * self.L * 1000 * self.i_d
        capital_cost_const = self.cost_b * self.L * 1000 * self.i_d

        # Annual energy cost = (energy_loss_factor / A) * energy_cost
        energy_cost_factor = energy_loss_factor * self.energy_cost

        # Total cost C = capital_cost_factor * A + capital_cost_const + energy_cost_factor / A
        # Minimize: dC/dA = capital_cost_factor - energy_cost_factor / A² = 0
        # A² = energy_cost_factor / capital_cost_factor

        A_optimal = math.sqrt(energy_cost_factor / capital_cost_factor)

        # Calculate costs at optimal cross-section
        R_optimal = resistance_factor / A_optimal
        power_loss = (self.I_max ** 2) * R_optimal / 1000  # in kW
        annual_energy_loss = power_loss * hours

        capital_cost = (self.cost_a * A_optimal + self.cost_b) * self.L * 1000
        annual_capital_cost = capital_cost * self.i_d
        annual_energy_cost = annual_energy_loss * self.energy_cost
        total_annual_cost = annual_capital_cost + annual_energy_cost

        return {
            'optimal_area': A_optimal,
            'resistance': R_optimal,
            'power_loss': power_loss,
            'annual_energy_loss': annual_energy_loss,
            'capital_cost': capital_cost,
            'annual_capital_cost': annual_capital_cost,
            'annual_energy_cost': annual_energy_cost,
            'total_annual_cost': total_annual_cost
        }

    def cost_vs_area(self, area_range):
        """Calculate total annual cost for a range of cross-sections"""
        hours = self.months * 30 * 24
        resistance_factor = self.rho * self.L * 2
        energy_loss_factor = (self.I_max ** 2) * resistance_factor * hours / 1000
        capital_cost_factor = self.cost_a * self.L * 1000 * self.i_d
        capital_cost_const = self.cost_b * self.L * 1000 * self.i_d
        energy_cost_factor = energy_loss_factor * self.energy_cost

        costs = []
        for A in area_range:
            if A > 0:
                annual_capital_cost = capital_cost_factor * A + capital_cost_const
                annual_energy_cost = energy_cost_factor / A
                total_cost = annual_capital_cost + annual_energy_cost
                costs.append(total_cost)
            else:
                costs.append(np.inf)

        return np.array(costs)


class ElectricalMachineSimulator:
    """Simulates electrical machines using differential equations"""

    @staticmethod
    def dc_motor_model(t, y, V, R, L, K_e, K_t, J, B):
        """
        DC Motor differential equations
        y = [i, omega] where i = current, omega = angular velocity
        V = applied voltage, R = armature resistance, L = inductance
        K_e = back EMF constant, K_t = torque constant
        J = moment of inertia, B = friction coefficient
        """
        i, omega = y
        di_dt = (V - R * i - K_e * omega) / L
        domega_dt = (K_t * i - B * omega) / J
        return np.array([di_dt, domega_dt])

    @staticmethod
    def rlc_circuit_model(t, y, V_source, R, L, C, omega_s):
        """
        RLC Circuit differential equations
        y = [i, v_c] where i = current, v_c = capacitor voltage
        V_source = source voltage amplitude, omega_s = source frequency
        """
        i, v_c = y
        V = V_source * np.sin(omega_s * t)
        di_dt = (V - v_c - R * i) / L
        dv_c_dt = i / C
        return np.array([di_dt, dv_c_dt])

    @staticmethod
    def transformer_inrush_model(t, y, V_peak, omega, R, L, flux_residual):
        """
        Transformer inrush current model
        y = [i, flux] where i = current, flux = magnetic flux
        """
        i, flux = y
        V = V_peak * np.sin(omega * t)
        di_dt = (V - R * i) / L
        dflux_dt = V
        return np.array([di_dt, dflux_dt])


class AdvancedElectricalEngineeringGUI:
    """Main GUI Application for Advanced Electrical Engineering Tools"""

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Electrical Engineering Simulator")
        self.root.geometry("1400x900")

        # Simulation control
        self.simulation_running = False
        self.animation_id = None

        # Configure root grid for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create main notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Create tabs
        self.create_feeder_cable_tab()
        self.create_dc_motor_tab()
        self.create_rlc_circuit_tab()
        self.create_power_system_tab()

        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky='ew')

        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)

    def on_window_resize(self, event):
        """Handle window resize events for auto-scaling"""
        if event.widget == self.root:
            # Update canvas sizes if needed
            pass

    def create_feeder_cable_tab(self):
        """Create feeder cable optimization tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Feeder Cable Optimizer")

        # Configure grid
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        # Input frame
        input_frame = ttk.LabelFrame(tab, text="Input Parameters", padding=10)
        input_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5, columnspan=2)

        # Default values from problem
        self.fc_voltage = self.create_input_field(input_frame, "Voltage (V):", 0, "500")
        self.fc_length = self.create_input_field(input_frame, "Length (km):", 1, "4")
        self.fc_current = self.create_input_field(input_frame, "Max Current (A):", 2, "200")
        self.fc_months = self.create_input_field(input_frame, "Operating Months/Year:", 3, "6")
        self.fc_resistance = self.create_input_field(input_frame, "Resistance (Ω/km/sq.cm):", 4, "0.17")
        self.fc_cost_a = self.create_input_field(input_frame, "Cable Cost Coeff. A (Rs/m):", 5, "120")
        self.fc_cost_b = self.create_input_field(input_frame, "Cable Cost Coeff. B (Rs/m):", 6, "24")
        self.fc_interest = self.create_input_field(input_frame, "Interest & Depreciation (%):", 7, "10")
        self.fc_energy_cost = self.create_input_field(input_frame, "Energy Cost (paise/kWh):", 8, "4")

        # Calculate button
        calc_btn = ttk.Button(input_frame, text="Calculate Optimal Cross-Section",
                             command=self.calculate_feeder_cable)
        calc_btn.grid(row=9, column=0, columnspan=2, pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(tab, text="Results", padding=10)
        results_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        self.fc_results = scrolledtext.ScrolledText(results_frame, width=50, height=25, wrap=tk.WORD)
        self.fc_results.pack(fill=tk.BOTH, expand=True)

        # Visualization frame
        viz_frame = ttk.LabelFrame(tab, text="Cost vs Cross-Section", padding=10)
        viz_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)

        self.fc_fig = Figure(figsize=(8, 6), dpi=100)
        self.fc_canvas = FigureCanvasTkAgg(self.fc_fig, master=viz_frame)
        self.fc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_dc_motor_tab(self):
        """Create DC motor simulation tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="DC Motor Dynamics")

        # Configure grid
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Input frame
        input_frame = ttk.LabelFrame(tab, text="DC Motor Parameters", padding=10)
        input_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        # Create parameter sliders
        self.dc_voltage = self.create_slider(input_frame, "Applied Voltage (V):", 0, 0, 500, 220)
        self.dc_resistance = self.create_slider(input_frame, "Armature Resistance (Ω):", 1, 0.1, 10, 2)
        self.dc_inductance = self.create_slider(input_frame, "Armature Inductance (H):", 2, 0.01, 1, 0.1)
        self.dc_ke = self.create_slider(input_frame, "Back EMF Constant (V·s/rad):", 3, 0.01, 1, 0.1)
        self.dc_kt = self.create_slider(input_frame, "Torque Constant (N·m/A):", 4, 0.01, 1, 0.1)
        self.dc_inertia = self.create_slider(input_frame, "Moment of Inertia (kg·m²):", 5, 0.001, 0.1, 0.01)
        self.dc_friction = self.create_slider(input_frame, "Friction Coefficient (N·m·s/rad):", 6, 0.001, 0.1, 0.01)

        # Control frame
        control_frame = ttk.Frame(tab)
        control_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

        ttk.Label(control_frame, text="ODE Solver:").pack(side=tk.LEFT, padx=5)
        self.dc_solver = ttk.Combobox(control_frame, values=["Euler", "RK45"], state='readonly', width=10)
        self.dc_solver.set("RK45")
        self.dc_solver.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Start", command=self.start_dc_motor_sim).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_dc_motor_sim).pack(side=tk.LEFT, padx=5)

        # Visualization frame
        viz_frame = ttk.LabelFrame(tab, text="Dynamic Response", padding=10)
        viz_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

        self.dc_fig = Figure(figsize=(12, 6), dpi=100)
        self.dc_canvas = FigureCanvasTkAgg(self.dc_fig, master=viz_frame)
        self.dc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_rlc_circuit_tab(self):
        """Create RLC circuit simulation tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="RLC Circuit Analysis")

        # Configure grid
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Input frame
        input_frame = ttk.LabelFrame(tab, text="RLC Circuit Parameters", padding=10)
        input_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        # Create parameter sliders
        self.rlc_voltage = self.create_slider(input_frame, "Source Voltage (V):", 0, 0, 500, 230)
        self.rlc_frequency = self.create_slider(input_frame, "Source Frequency (Hz):", 1, 10, 1000, 50)
        self.rlc_resistance = self.create_slider(input_frame, "Resistance (Ω):", 2, 1, 1000, 100)
        self.rlc_inductance = self.create_slider(input_frame, "Inductance (mH):", 3, 1, 1000, 100)
        self.rlc_capacitance = self.create_slider(input_frame, "Capacitance (μF):", 4, 1, 1000, 100)

        # Control frame
        control_frame = ttk.Frame(tab)
        control_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

        ttk.Label(control_frame, text="ODE Solver:").pack(side=tk.LEFT, padx=5)
        self.rlc_solver = ttk.Combobox(control_frame, values=["Euler", "RK45"], state='readonly', width=10)
        self.rlc_solver.set("RK45")
        self.rlc_solver.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Start", command=self.start_rlc_sim).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_rlc_sim).pack(side=tk.LEFT, padx=5)

        # Visualization frame
        viz_frame = ttk.LabelFrame(tab, text="Circuit Response", padding=10)
        viz_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

        self.rlc_fig = Figure(figsize=(12, 6), dpi=100)
        self.rlc_canvas = FigureCanvasTkAgg(self.rlc_fig, master=viz_frame)
        self.rlc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_power_system_tab(self):
        """Create power system analysis tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Power System Analysis")

        # Configure grid
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Input frame
        input_frame = ttk.LabelFrame(tab, text="Three-Phase Power Analysis", padding=10)
        input_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        self.ps_voltage = self.create_input_field(input_frame, "Line Voltage (V):", 0, "400")
        self.ps_current = self.create_input_field(input_frame, "Line Current (A):", 1, "100")
        self.ps_pf = self.create_input_field(input_frame, "Power Factor:", 2, "0.85")

        ttk.Button(input_frame, text="Calculate Power", command=self.calculate_power).grid(
            row=3, column=0, columnspan=2, pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(tab, text="Power Analysis Results", padding=10)
        results_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        self.ps_results = scrolledtext.ScrolledText(results_frame, width=60, height=20, wrap=tk.WORD)
        self.ps_results.pack(fill=tk.BOTH, expand=True)

    def create_input_field(self, parent, label_text, row, default_value):
        """Create a labeled input field"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=5)
        entry = ttk.Entry(parent, width=20)
        entry.grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        entry.insert(0, default_value)
        parent.grid_columnconfigure(1, weight=1)
        return entry

    def create_slider(self, parent, label_text, row, min_val, max_val, default_val):
        """Create a labeled slider with value display"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=5)

        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=1, sticky='ew', padx=5, pady=5)
        slider_frame.grid_columnconfigure(0, weight=1)

        var = tk.DoubleVar(value=default_val)

        slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                          variable=var, command=lambda v: value_label.config(text=f"{float(v):.4f}"))
        slider.grid(row=0, column=0, sticky='ew')

        value_label = ttk.Label(slider_frame, text=f"{default_val:.4f}", width=10)
        value_label.grid(row=0, column=1, padx=5)

        parent.grid_columnconfigure(1, weight=1)
        return var

    def calculate_feeder_cable(self):
        """Calculate optimal feeder cable cross-section"""
        try:
            # Get input values
            voltage = float(self.fc_voltage.get())
            length = float(self.fc_length.get())
            current = float(self.fc_current.get())
            months = float(self.fc_months.get())
            resistance = float(self.fc_resistance.get())
            cost_a = float(self.fc_cost_a.get())
            cost_b = float(self.fc_cost_b.get())
            interest = float(self.fc_interest.get()) / 100
            energy_cost = float(self.fc_energy_cost.get()) / 100  # Convert paise to rupees

            # Create optimizer
            optimizer = FeederCableOptimizer(voltage, length, current, months,
                                            resistance, cost_a, cost_b,
                                            interest, energy_cost)

            # Calculate optimal cross-section
            results = optimizer.calculate_optimal_cross_section()

            # Display results
            self.fc_results.delete(1.0, tk.END)
            self.fc_results.insert(tk.END, "=" * 60 + "\n")
            self.fc_results.insert(tk.END, "FEEDER CABLE OPTIMIZATION RESULTS\n")
            self.fc_results.insert(tk.END, "=" * 60 + "\n\n")

            self.fc_results.insert(tk.END, "INPUT PARAMETERS:\n")
            self.fc_results.insert(tk.END, f"  Voltage: {voltage} V\n")
            self.fc_results.insert(tk.END, f"  Cable Length: {length} km\n")
            self.fc_results.insert(tk.END, f"  Maximum Current: {current} A\n")
            self.fc_results.insert(tk.END, f"  Operating Time: {months} months/year\n")
            self.fc_results.insert(tk.END, f"  Resistance: {resistance} Ω/km/sq.cm\n")
            self.fc_results.insert(tk.END, f"  Cable Cost: Rs. ({cost_a}A + {cost_b})/meter\n")
            self.fc_results.insert(tk.END, f"  Interest & Depreciation: {interest*100}%\n")
            self.fc_results.insert(tk.END, f"  Energy Cost: {energy_cost*100} paise/kWh\n\n")

            self.fc_results.insert(tk.END, "OPTIMAL DESIGN:\n")
            self.fc_results.insert(tk.END, f"  Optimal Cross-Section: {results['optimal_area']:.4f} sq.cm\n")
            self.fc_results.insert(tk.END, f"  Cable Resistance: {results['resistance']:.6f} Ω\n")
            self.fc_results.insert(tk.END, f"  Power Loss at Full Load: {results['power_loss']:.2f} kW\n\n")

            self.fc_results.insert(tk.END, "ANNUAL COSTS:\n")
            self.fc_results.insert(tk.END, f"  Capital Cost: Rs. {results['capital_cost']:.2f}\n")
            self.fc_results.insert(tk.END, f"  Annual Capital Charges: Rs. {results['annual_capital_cost']:.2f}\n")
            self.fc_results.insert(tk.END, f"  Annual Energy Loss: {results['annual_energy_loss']:.2f} kWh\n")
            self.fc_results.insert(tk.END, f"  Annual Energy Cost: Rs. {results['annual_energy_cost']:.2f}\n")
            self.fc_results.insert(tk.END, f"  Total Annual Cost: Rs. {results['total_annual_cost']:.2f}\n\n")

            self.fc_results.insert(tk.END, "=" * 60 + "\n")

            # Plot cost vs area
            self.fc_fig.clear()
            ax = self.fc_fig.add_subplot(111)

            # Generate area range around optimal
            A_opt = results['optimal_area']
            area_range = np.linspace(max(0.1, A_opt * 0.3), A_opt * 2.5, 100)
            costs = optimizer.cost_vs_area(area_range)

            ax.plot(area_range, costs, 'b-', linewidth=2, label='Total Annual Cost')
            ax.axvline(A_opt, color='r', linestyle='--', linewidth=2, label=f'Optimal: {A_opt:.4f} sq.cm')
            ax.scatter([A_opt], [results['total_annual_cost']], color='r', s=100, zorder=5)

            ax.set_xlabel('Cross-Sectional Area (sq.cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Annual Cost (Rs.)', fontsize=12, fontweight='bold')
            ax.set_title('Economic Analysis: Cost vs Cross-Section', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            self.fc_fig.tight_layout()
            self.fc_canvas.draw()

            self.status_bar.config(text=f"Calculation complete. Optimal area: {A_opt:.4f} sq.cm")

        except Exception as e:
            messagebox.showerror("Error", f"Calculation error: {str(e)}")
            self.status_bar.config(text="Error in calculation")

    def start_dc_motor_sim(self):
        """Start DC motor simulation"""
        try:
            # Get parameters
            V = self.dc_voltage.get()
            R = self.dc_resistance.get()
            L = self.dc_inductance.get()
            K_e = self.dc_ke.get()
            K_t = self.dc_kt.get()
            J = self.dc_inertia.get()
            B = self.dc_friction.get()
            solver_type = self.dc_solver.get()

            # Time array
            t = np.linspace(0, 2, 1000)  # 2 seconds simulation

            # Initial conditions [current, angular_velocity]
            y0 = np.array([0.0, 0.0])

            # Define ODE function
            def motor_ode(t, y):
                return ElectricalMachineSimulator.dc_motor_model(t, y, V, R, L, K_e, K_t, J, B)

            # Solve ODE
            if solver_type == "Euler":
                solution = ODESolver.euler(motor_ode, y0, t)
            else:
                solution = ODESolver.rk45(motor_ode, y0, t)

            current = solution[:, 0]
            omega = solution[:, 1]

            # Calculate torque and power
            torque = K_t * current
            power = torque * omega

            # Plot results
            self.dc_fig.clear()

            ax1 = self.dc_fig.add_subplot(2, 2, 1)
            ax1.plot(t, current, 'b-', linewidth=2)
            ax1.set_xlabel('Time (s)', fontweight='bold')
            ax1.set_ylabel('Current (A)', fontweight='bold')
            ax1.set_title('Armature Current vs Time', fontweight='bold')
            ax1.grid(True, alpha=0.3)

            ax2 = self.dc_fig.add_subplot(2, 2, 2)
            ax2.plot(t, omega * 60 / (2 * np.pi), 'r-', linewidth=2)  # Convert to RPM
            ax2.set_xlabel('Time (s)', fontweight='bold')
            ax2.set_ylabel('Speed (RPM)', fontweight='bold')
            ax2.set_title('Motor Speed vs Time', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            ax3 = self.dc_fig.add_subplot(2, 2, 3)
            ax3.plot(t, torque, 'g-', linewidth=2)
            ax3.set_xlabel('Time (s)', fontweight='bold')
            ax3.set_ylabel('Torque (N·m)', fontweight='bold')
            ax3.set_title('Motor Torque vs Time', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            ax4 = self.dc_fig.add_subplot(2, 2, 4)
            ax4.plot(t, power, 'm-', linewidth=2)
            ax4.set_xlabel('Time (s)', fontweight='bold')
            ax4.set_ylabel('Power (W)', fontweight='bold')
            ax4.set_title('Mechanical Power vs Time', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            self.dc_fig.tight_layout()
            self.dc_canvas.draw()

            self.status_bar.config(text=f"DC Motor simulation complete using {solver_type} method")

        except Exception as e:
            messagebox.showerror("Error", f"Simulation error: {str(e)}")
            self.status_bar.config(text="Error in simulation")

    def reset_dc_motor_sim(self):
        """Reset DC motor simulation"""
        self.dc_fig.clear()
        self.dc_canvas.draw()
        self.status_bar.config(text="DC Motor simulation reset")

    def start_rlc_sim(self):
        """Start RLC circuit simulation"""
        try:
            # Get parameters
            V_source = self.rlc_voltage.get()
            f = self.rlc_frequency.get()
            R = self.rlc_resistance.get()
            L = self.rlc_inductance.get() / 1000  # Convert mH to H
            C = self.rlc_capacitance.get() * 1e-6  # Convert μF to F
            omega_s = 2 * np.pi * f
            solver_type = self.rlc_solver.get()

            # Time array - multiple cycles
            t = np.linspace(0, 0.1, 5000)

            # Initial conditions [current, capacitor_voltage]
            y0 = np.array([0.0, 0.0])

            # Define ODE function
            def rlc_ode(t, y):
                return ElectricalMachineSimulator.rlc_circuit_model(t, y, V_source, R, L, C, omega_s)

            # Solve ODE
            if solver_type == "Euler":
                solution = ODESolver.euler(rlc_ode, y0, t)
            else:
                solution = ODESolver.rk45(rlc_ode, y0, t)

            current = solution[:, 0]
            v_capacitor = solution[:, 1]
            v_source = V_source * np.sin(omega_s * t)
            v_resistor = R * current
            v_inductor = L * np.gradient(current, t)

            # Calculate resonance frequency and quality factor
            f_resonance = 1 / (2 * np.pi * np.sqrt(L * C))
            Q_factor = (1 / R) * np.sqrt(L / C)

            # Plot results
            self.rlc_fig.clear()

            ax1 = self.rlc_fig.add_subplot(2, 2, 1)
            ax1.plot(t * 1000, v_source, 'b-', linewidth=2, label='Source')
            ax1.plot(t * 1000, v_capacitor, 'r-', linewidth=2, label='Capacitor')
            ax1.plot(t * 1000, v_resistor, 'g-', linewidth=2, label='Resistor')
            ax1.set_xlabel('Time (ms)', fontweight='bold')
            ax1.set_ylabel('Voltage (V)', fontweight='bold')
            ax1.set_title('Voltages vs Time', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=8)

            ax2 = self.rlc_fig.add_subplot(2, 2, 2)
            ax2.plot(t * 1000, current, 'b-', linewidth=2)
            ax2.set_xlabel('Time (ms)', fontweight='bold')
            ax2.set_ylabel('Current (A)', fontweight='bold')
            ax2.set_title('Circuit Current vs Time', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            ax3 = self.rlc_fig.add_subplot(2, 2, 3)
            ax3.plot(v_source, current, 'b-', linewidth=1, alpha=0.7)
            ax3.set_xlabel('Voltage (V)', fontweight='bold')
            ax3.set_ylabel('Current (A)', fontweight='bold')
            ax3.set_title('V-I Characteristic', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Frequency response
            ax4 = self.rlc_fig.add_subplot(2, 2, 4)
            freq_range = np.logspace(0, 4, 500)  # 1 Hz to 10 kHz
            impedance = []
            for freq in freq_range:
                omega = 2 * np.pi * freq
                Z = np.sqrt(R**2 + (omega * L - 1 / (omega * C))**2)
                impedance.append(Z)

            ax4.semilogx(freq_range, impedance, 'r-', linewidth=2)
            ax4.axvline(f_resonance, color='b', linestyle='--', linewidth=2,
                       label=f'Resonance: {f_resonance:.2f} Hz')
            ax4.set_xlabel('Frequency (Hz)', fontweight='bold')
            ax4.set_ylabel('Impedance (Ω)', fontweight='bold')
            ax4.set_title(f'Frequency Response (Q = {Q_factor:.2f})', fontweight='bold')
            ax4.grid(True, alpha=0.3, which='both')
            ax4.legend(loc='best', fontsize=8)

            self.rlc_fig.tight_layout()
            self.rlc_canvas.draw()

            self.status_bar.config(text=f"RLC simulation complete. Resonance: {f_resonance:.2f} Hz, Q: {Q_factor:.2f}")

        except Exception as e:
            messagebox.showerror("Error", f"Simulation error: {str(e)}")
            self.status_bar.config(text="Error in simulation")

    def reset_rlc_sim(self):
        """Reset RLC circuit simulation"""
        self.rlc_fig.clear()
        self.rlc_canvas.draw()
        self.status_bar.config(text="RLC circuit simulation reset")

    def stop_simulation(self):
        """Stop any running simulation"""
        self.simulation_running = False
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
        self.status_bar.config(text="Simulation stopped")

    def calculate_power(self):
        """Calculate three-phase power"""
        try:
            V_line = float(self.ps_voltage.get())
            I_line = float(self.ps_current.get())
            pf = float(self.ps_pf.get())

            # Three-phase calculations
            sqrt3 = np.sqrt(3)
            V_phase = V_line / sqrt3

            # Power calculations
            P_apparent = sqrt3 * V_line * I_line  # VA
            P_real = P_apparent * pf  # W
            P_reactive = P_apparent * np.sqrt(1 - pf**2)  # VAR

            # Current components
            I_real = I_line * pf
            I_reactive = I_line * np.sqrt(1 - pf**2)

            # Display results
            self.ps_results.delete(1.0, tk.END)
            self.ps_results.insert(tk.END, "=" * 70 + "\n")
            self.ps_results.insert(tk.END, "THREE-PHASE POWER SYSTEM ANALYSIS\n")
            self.ps_results.insert(tk.END, "=" * 70 + "\n\n")

            self.ps_results.insert(tk.END, "INPUT PARAMETERS:\n")
            self.ps_results.insert(tk.END, f"  Line Voltage (V_L): {V_line:.2f} V\n")
            self.ps_results.insert(tk.END, f"  Line Current (I_L): {I_line:.2f} A\n")
            self.ps_results.insert(tk.END, f"  Power Factor (cos φ): {pf:.4f}\n")
            self.ps_results.insert(tk.END, f"  Power Factor Angle (φ): {np.arccos(pf) * 180 / np.pi:.2f}°\n\n")

            self.ps_results.insert(tk.END, "VOLTAGE AND CURRENT:\n")
            self.ps_results.insert(tk.END, f"  Phase Voltage (V_ph): {V_phase:.2f} V\n")
            self.ps_results.insert(tk.END, f"  Active Current Component: {I_real:.2f} A\n")
            self.ps_results.insert(tk.END, f"  Reactive Current Component: {I_reactive:.2f} A\n\n")

            self.ps_results.insert(tk.END, "POWER CALCULATIONS:\n")
            self.ps_results.insert(tk.END, f"  Apparent Power (S): {P_apparent:.2f} VA = {P_apparent/1000:.2f} kVA\n")
            self.ps_results.insert(tk.END, f"  Real Power (P): {P_real:.2f} W = {P_real/1000:.2f} kW\n")
            self.ps_results.insert(tk.END, f"  Reactive Power (Q): {P_reactive:.2f} VAR = {P_reactive/1000:.2f} kVAR\n\n")

            self.ps_results.insert(tk.END, "EFFICIENCY METRICS:\n")
            if P_apparent > 0:
                efficiency = (P_real / P_apparent) * 100
                self.ps_results.insert(tk.END, f"  Power Factor (%): {efficiency:.2f}%\n")

            # Capacitor for power factor correction
            target_pf = 0.95
            if pf < target_pf:
                Q_correction = P_real * (np.tan(np.arccos(pf)) - np.tan(np.arccos(target_pf)))
                C_required = Q_correction / (2 * np.pi * 50 * V_line**2)  # Assuming 50 Hz

                self.ps_results.insert(tk.END, f"\nPOWER FACTOR CORRECTION (to {target_pf}):\n")
                self.ps_results.insert(tk.END, f"  Required Capacitor Bank: {Q_correction:.2f} VAR = {Q_correction/1000:.2f} kVAR\n")
                self.ps_results.insert(tk.END, f"  Capacitance per Phase (50 Hz): {C_required*1e6:.2f} μF\n")

            self.ps_results.insert(tk.END, "\n" + "=" * 70 + "\n")

            self.status_bar.config(text=f"Power calculation complete. Real Power: {P_real/1000:.2f} kW")

        except Exception as e:
            messagebox.showerror("Error", f"Calculation error: {str(e)}")
            self.status_bar.config(text="Error in calculation")


def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = AdvancedElectricalEngineeringGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
