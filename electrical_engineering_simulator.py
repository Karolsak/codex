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


class CompoundGeneratorCalculator:
    """Calculate compound generator parameters"""

    def __init__(self, power_kw, voltage_v, speed_rpm, poles, diameter_m, length_m, slot_pitch_m, flux_wb):
        """
        Initialize compound generator calculator

        Args:
            power_kw: Power rating in kW
            voltage_v: Voltage in V
            speed_rpm: Speed in rpm
            poles: Number of poles
            diameter_m: External diameter of armature in m
            length_m: Gross armature length in m
            slot_pitch_m: Armature slot pitch in m
            flux_wb: Flux per pole in Wb
        """
        self.power_kw = power_kw
        self.voltage_v = voltage_v
        self.speed_rpm = speed_rpm
        self.poles = poles
        self.diameter_m = diameter_m
        self.length_m = length_m
        self.slot_pitch_m = slot_pitch_m
        self.flux_wb = flux_wb
        
        self.results = {}
        self.calculate()

    def calculate(self):
        """Perform all generator calculations"""
        # EMF equation: E = (φ × Z × N × P) / (60 × A)
        # Where: E = EMF, φ = flux per pole, Z = number of conductors
        # N = speed in rpm, P = number of poles, A = number of parallel paths
        
        # For compound generator (lap or wave winding)
        # Assume wave winding: A = 2
        # Assume lap winding: A = P
        
        # Generated EMF (assuming 5% voltage drop)
        emf = self.voltage_v * 1.05  # Account for voltage drop
        
        # Calculate number of conductors using EMF equation
        # E = (φ × Z × N × P) / (60 × A)
        # For wave winding: A = 2
        A_wave = 2
        Z_wave = (emf * 60 * A_wave) / (self.flux_wb * self.speed_rpm * self.poles)
        
        # For lap winding: A = P
        A_lap = self.poles
        Z_lap = (emf * 60 * A_lap) / (self.flux_wb * self.speed_rpm * self.poles)
        
        # Choose wave winding as it's more common for high voltage machines
        Z = Z_wave
        A = A_wave
        
        # Round to nearest even number (conductors come in pairs)
        Z = int(np.round(Z / 2) * 2)
        
        # Calculate number of slots
        # Circumference of armature
        circumference = np.pi * self.diameter_m
        
        # Number of slots = Circumference / slot pitch
        num_slots = int(np.round(circumference / self.slot_pitch_m))
        
        # Conductors per slot
        conductors_per_slot = Z / num_slots
        
        # Calculate armature current
        # Power = Voltage × Current
        armature_current = (self.power_kw * 1000) / self.voltage_v
        
        # Calculate conductor cross-sectional area
        # Current density for copper: typically 4-6 A/mm²
        current_density = 5.0  # A/mm²
        current_per_conductor = armature_current / (Z / 2)  # Z/2 parallel paths for wave
        conductor_area_mm2 = current_per_conductor / current_density
        
        # Calculate resistance of armature winding
        # Mean length of turn
        pole_pitch = (np.pi * self.diameter_m) / self.poles
        mean_length_turn = 2 * (self.length_m + pole_pitch)
        
        # Total length of conductor
        total_length = mean_length_turn * Z
        
        # Resistance of copper at 75°C
        # Resistivity of copper at 75°C ≈ 0.021 Ω·mm²/m
        resistivity_copper = 0.021  # Ω·mm²/m
        
        # Resistance = (ρ × length) / area
        conductor_resistance = (resistivity_copper * total_length) / conductor_area_mm2
        
        # For wave winding with 2 parallel paths
        armature_resistance = conductor_resistance / (A / 2)
        
        # Calculate specific electric loading
        # q = (total ampere-conductors) / (armature periphery)
        specific_electric_loading = (armature_current * Z) / circumference
        
        # Calculate specific magnetic loading
        # Average flux density in air gap
        avg_flux_density = self.flux_wb / (pole_pitch * self.length_m)
        
        # Store results
        self.results = {
            'emf': emf,
            'num_conductors': int(Z),
            'num_slots': num_slots,
            'conductors_per_slot': conductors_per_slot,
            'armature_resistance': armature_resistance,
            'armature_current': armature_current,
            'winding_type': 'Wave',
            'parallel_paths': A,
            'conductor_area_mm2': conductor_area_mm2,
            'mean_length_turn': mean_length_turn,
            'total_conductor_length': total_length,
            'pole_pitch': pole_pitch,
            'circumference': circumference,
            'specific_electric_loading': specific_electric_loading,
            'avg_flux_density': avg_flux_density,
            'current_per_conductor': current_per_conductor,
            'current_density': current_density
        }
        
        return self.results


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

        # Create GUI
        self.create_widgets()

        # Bind resize event
        self.bind('<Configure>', self.on_resize)

    def create_widgets(self):
        """Create all GUI widgets"""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.create_compound_generator_tab()
        self.create_tariff_tab()
        self.create_dynamics_tab()
        self.create_rlc_tab()

    def create_compound_generator_tab(self):
        """Create compound generator calculator tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Compound Generator")

        # Main container with two panels
        main_frame = ttk.Frame(tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Inputs
        left_frame = ttk.LabelFrame(main_frame, text="Generator Parameters", padding=10)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Right panel - Results
        right_frame = ttk.LabelFrame(main_frame, text="Calculation Results", padding=10)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # Input fields with sliders
        row = 0

        # Power
        ttk.Label(left_frame, text="Power Rating (kW):").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_power_var = tk.DoubleVar(value=500.0)
        power_frame = ttk.Frame(left_frame)
        power_frame.grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Entry(power_frame, textvariable=self.gen_power_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Scale(power_frame, from_=100, to=2000, variable=self.gen_power_var, 
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=2)
        row += 1

        # Voltage
        ttk.Label(left_frame, text="Voltage (V):").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_voltage_var = tk.DoubleVar(value=440.0)
        voltage_frame = ttk.Frame(left_frame)
        voltage_frame.grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Entry(voltage_frame, textvariable=self.gen_voltage_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Scale(voltage_frame, from_=220, to=1000, variable=self.gen_voltage_var,
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=2)
        row += 1

        # Speed
        ttk.Label(left_frame, text="Speed (rpm):").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_speed_var = tk.DoubleVar(value=375.0)
        speed_frame = ttk.Frame(left_frame)
        speed_frame.grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Entry(speed_frame, textvariable=self.gen_speed_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Scale(speed_frame, from_=100, to=1500, variable=self.gen_speed_var,
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=2)
        row += 1

        # Number of poles
        ttk.Label(left_frame, text="Number of Poles:").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_poles_var = tk.IntVar(value=8)
        poles_frame = ttk.Frame(left_frame)
        poles_frame.grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Entry(poles_frame, textvariable=self.gen_poles_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Scale(poles_frame, from_=2, to=16, variable=self.gen_poles_var,
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=2)
        row += 1

        # Armature diameter
        ttk.Label(left_frame, text="Armature Diameter (m):").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_diameter_var = tk.DoubleVar(value=1.1)
        diameter_frame = ttk.Frame(left_frame)
        diameter_frame.grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Entry(diameter_frame, textvariable=self.gen_diameter_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Scale(diameter_frame, from_=0.3, to=3.0, variable=self.gen_diameter_var,
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=2)
        row += 1

        # Armature length
        ttk.Label(left_frame, text="Armature Length (m):").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_length_var = tk.DoubleVar(value=0.3)
        length_frame = ttk.Frame(left_frame)
        length_frame.grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Entry(length_frame, textvariable=self.gen_length_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Scale(length_frame, from_=0.1, to=1.0, variable=self.gen_length_var,
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=2)
        row += 1

        # Slot pitch
        ttk.Label(left_frame, text="Slot Pitch (m):").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_slot_pitch_var = tk.DoubleVar(value=0.025)
        slot_pitch_frame = ttk.Frame(left_frame)
        slot_pitch_frame.grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Entry(slot_pitch_frame, textvariable=self.gen_slot_pitch_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Scale(slot_pitch_frame, from_=0.01, to=0.1, variable=self.gen_slot_pitch_var,
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=2)
        row += 1

        # Flux per pole
        ttk.Label(left_frame, text="Flux per Pole (Wb):").grid(row=row, column=0, sticky='w', pady=5)
        self.gen_flux_var = tk.DoubleVar(value=0.0875)
        flux_frame = ttk.Frame(left_frame)
        flux_frame.grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Entry(flux_frame, textvariable=self.gen_flux_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Scale(flux_frame, from_=0.01, to=0.2, variable=self.gen_flux_var,
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=2)
        row += 1

        # Calculate button
        ttk.Button(left_frame, text="Calculate Parameters", command=self.calculate_generator,
                   style='Accent.TButton').grid(row=row, column=0, columnspan=2, pady=20)
        row += 1

        # Reset button
        ttk.Button(left_frame, text="Reset to Default", command=self.reset_generator_params).grid(
            row=row, column=0, columnspan=2, pady=5)

        # Results text widget
        self.gen_results_text = tk.Text(right_frame, wrap=tk.WORD, font=('Courier', 10))
        self.gen_results_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(right_frame, command=self.gen_results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gen_results_text.config(yscrollcommand=scrollbar.set)

        # Visualization frame at bottom
        viz_frame = ttk.LabelFrame(tab, text="Generator Visualization", padding=10)
        viz_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create matplotlib figure for visualization
        self.fig_gen = Figure(figsize=(12, 4))
        self.ax_gen1 = self.fig_gen.add_subplot(1, 3, 1)
        self.ax_gen2 = self.fig_gen.add_subplot(1, 3, 2)
        self.ax_gen3 = self.fig_gen.add_subplot(1, 3, 3)

        self.canvas_gen = FigureCanvasTkAgg(self.fig_gen, master=viz_frame)
        self.canvas_gen.draw()
        self.canvas_gen.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def calculate_generator(self):
        """Calculate and display generator parameters"""
        try:
            # Get input values
            power = self.gen_power_var.get()
            voltage = self.gen_voltage_var.get()
            speed = self.gen_speed_var.get()
            poles = self.gen_poles_var.get()
            diameter = self.gen_diameter_var.get()
            length = self.gen_length_var.get()
            slot_pitch = self.gen_slot_pitch_var.get()
            flux = self.gen_flux_var.get()

            # Calculate generator parameters
            calculator = CompoundGeneratorCalculator(power, voltage, speed, poles,
                                                    diameter, length, slot_pitch, flux)
            results = calculator.results

            # Display results
            self.gen_results_text.delete(1.0, tk.END)

            output = "=" * 75 + "\n"
            output += "COMPOUND GENERATOR CALCULATION RESULTS\n"
            output += "=" * 75 + "\n\n"

            output += "INPUT PARAMETERS:\n"
            output += "-" * 75 + "\n"
            output += f"Power Rating:                  {power:>15.2f} kW\n"
            output += f"Voltage:                       {voltage:>15.2f} V\n"
            output += f"Speed:                         {speed:>15.2f} rpm\n"
            output += f"Number of Poles:               {poles:>15d}\n"
            output += f"Armature Diameter:             {diameter:>15.3f} m\n"
            output += f"Armature Length:               {length:>15.3f} m\n"
            output += f"Slot Pitch:                    {slot_pitch:>15.4f} m ({slot_pitch*100:.2f} cm)\n"
            output += f"Flux per Pole:                 {flux:>15.4f} Wb\n\n"

            output += "=" * 75 + "\n"
            output += "ARMATURE WINDING DESIGN:\n"
            output += "=" * 75 + "\n"
            output += f"Generated EMF:                 {results['emf']:>15.2f} V\n"
            output += f"Winding Type:                  {results['winding_type']:>15s}\n"
            output += f"Number of Parallel Paths:      {results['parallel_paths']:>15d}\n\n"

            output += "*** PRIMARY RESULTS ***\n"
            output += f"Number of Conductors (Z):      {results['num_conductors']:>15d}\n"
            output += f"Number of Slots:               {results['num_slots']:>15d}\n"
            output += f"Conductors per Slot:           {results['conductors_per_slot']:>15.2f}\n"
            output += f"Armature Resistance (Ra):      {results['armature_resistance']:>15.4f} Ω\n\n"

            output += "=" * 75 + "\n"
            output += "DETAILED CALCULATIONS:\n"
            output += "=" * 75 + "\n"
            output += f"Armature Current:              {results['armature_current']:>15.2f} A\n"
            output += f"Current per Conductor:         {results['current_per_conductor']:>15.2f} A\n"
            output += f"Current Density:               {results['current_density']:>15.2f} A/mm²\n"
            output += f"Conductor Area:                {results['conductor_area_mm2']:>15.2f} mm²\n\n"

            output += "GEOMETRIC PARAMETERS:\n"
            output += "-" * 75 + "\n"
            output += f"Pole Pitch:                    {results['pole_pitch']:>15.4f} m\n"
            output += f"Armature Circumference:        {results['circumference']:>15.4f} m\n"
            output += f"Mean Length of Turn:           {results['mean_length_turn']:>15.4f} m\n"
            output += f"Total Conductor Length:        {results['total_conductor_length']:>15.2f} m\n\n"

            output += "LOADING PARAMETERS:\n"
            output += "-" * 75 + "\n"
            output += f"Specific Electric Loading:     {results['specific_electric_loading']:>15.2f} A/m\n"
            output += f"Average Flux Density:          {results['avg_flux_density']:>15.4f} T\n\n"

            output += "=" * 75 + "\n"
            output += "FORMULAS USED:\n"
            output += "=" * 75 + "\n"
            output += "1. EMF Equation: E = (φ × Z × N × P) / (60 × A)\n"
            output += "2. Number of Slots = Circumference / Slot Pitch\n"
            output += "3. Armature Current = Power / Voltage\n"
            output += "4. Resistance = (ρ × Length) / Area\n"
            output += "5. Specific Electric Loading = (I × Z) / Circumference\n\n"

            self.gen_results_text.insert(1.0, output)

            # Update visualizations
            self.update_generator_plots(results)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid numeric values: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def update_generator_plots(self, results):
        """Update generator visualization plots"""
        # Clear previous plots
        self.ax_gen1.clear()
        self.ax_gen2.clear()
        self.ax_gen3.clear()

        # Plot 1: Bar chart of key parameters
        params = ['Conductors', 'Slots', 'Poles']
        values = [results['num_conductors'], results['num_slots'], self.gen_poles_var.get()]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = self.ax_gen1.bar(params, values, color=colors, alpha=0.7, edgecolor='black')
        self.ax_gen1.set_ylabel('Count')
        self.ax_gen1.set_title('Armature Configuration', fontweight='bold')
        self.ax_gen1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            self.ax_gen1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Current and resistance
        self.ax_gen2.barh(['Armature\nCurrent (A)', 'Armature\nResistance (Ω)'],
                         [results['armature_current'], results['armature_resistance']*100],
                         color=['#d62728', '#9467bd'], alpha=0.7, edgecolor='black')
        self.ax_gen2.set_xlabel('Value')
        self.ax_gen2.set_title('Electrical Parameters', fontweight='bold')
        self.ax_gen2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (val, label) in enumerate(zip([results['armature_current'], 
                                               results['armature_resistance']],
                                             ['A', 'Ω'])):
            self.ax_gen2.text(val if label == 'A' else val*100, i,
                            f'  {val:.2f} {label}', va='center', fontweight='bold')

        # Plot 3: Pie chart of winding distribution
        labels = ['Active\nConductors', 'End\nConnections']
        active_length = results['total_conductor_length'] * (self.gen_length_var.get() / 
                       (self.gen_length_var.get() + results['pole_pitch']))
        end_length = results['total_conductor_length'] - active_length
        sizes = [active_length, end_length]
        colors_pie = ['#8c564b', '#e377c2']
        
        wedges, texts, autotexts = self.ax_gen3.pie(sizes, labels=labels, colors=colors_pie,
                                                     autopct='%1.1f%%', startangle=90,
                                                     textprops={'fontweight': 'bold'})
        self.ax_gen3.set_title('Conductor Length Distribution', fontweight='bold')

        self.fig_gen.tight_layout()
        self.canvas_gen.draw()

    def reset_generator_params(self):
        """Reset generator parameters to default values"""
        self.gen_power_var.set(500.0)
        self.gen_voltage_var.set(440.0)
        self.gen_speed_var.set(375.0)
        self.gen_poles_var.set(8)
        self.gen_diameter_var.set(1.1)
        self.gen_length_var.set(0.3)
        self.gen_slot_pitch_var.set(0.025)
        self.gen_flux_var.set(0.0875)

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

        # Solver selection
        ttk.Label(params_frame, text="ODE Solver:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.solver_var = tk.StringVar(value="RK45")
        solver_combo = ttk.Combobox(params_frame, textvariable=self.solver_var,
                                     values=["RK45", "Euler"], state='readonly', width=18)
        solver_combo.grid(row=4, column=1, padx=5, pady=5, sticky='w')

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

        # Time parameters
        t_span = (0, 10)
        dt = 0.01

        # Solve ODE
        solver = ODESolver()
        if self.solver_var.get() == "RK45":
            t, y = solver.rk45(dynamics.swing_equation, y0, t_span, dt)
        else:
            t, y = solver.euler(dynamics.swing_equation, y0, t_span, dt)

        # Convert delta to degrees
        delta_deg = np.rad2deg(y[:, 0])
        omega = y[:, 1]

        # Update plots
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(t, delta_deg, 'b-', linewidth=2)
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Rotor Angle δ (deg)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title(f'Generator Rotor Angle (Swing Equation) - Solver: {self.solver_var.get()}')

        self.ax2.plot(t, omega, 'r-', linewidth=2)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Speed Deviation Δω (rad/s)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_title('Rotor Speed Deviation')

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
        # This allows automatic scaling of the GUI
        pass


def main():
    """Main entry point"""
    app = ElectricalEngineeringSimulator()
    app.mainloop()


if __name__ == "__main__":
    main()
