"""
Motor Analysis Core - Economic and Dynamic Analysis without GUI
"""

import numpy as np


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


if __name__ == "__main__":
    from datetime import datetime

    print("=" * 80)
    print("MOTOR REPLACEMENT ECONOMIC ANALYSIS")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run economic analysis
    analysis = MotorEconomicAnalysis()
    results = analysis.analyze_motors()

    # Display results
    print("-" * 80)
    print("MOTOR A - ANALYSIS")
    print("-" * 80)
    motor_a = results['Motor A']
    print(f"Initial Cost:              Rs. {motor_a['initial_cost']:,.2f}")
    print(f"Efficiency (Full Load):    {analysis.motor_a_eta_full * 100:.2f}%")
    print(f"Efficiency (Half Load):    {analysis.motor_a_eta_half * 100:.2f}%")
    print(f"\nAnnual Operating Costs:")
    print(f"  Energy Consumption:      {motor_a['annual_energy_kwh']:,.2f} kWh")
    print(f"  Energy Cost:             Rs. {motor_a['annual_energy_cost']:,.2f}")
    print(f"  Maintenance Cost:        Rs. {motor_a['annual_maintenance']:,.2f}")
    print(f"  Total Annual Cost:       Rs. {motor_a['total_annual_cost']:,.2f}")
    print(f"\nPresent Worth Analysis:")
    print(f"  PW of Operating Costs:   Rs. {motor_a['pw_operating']:,.2f}")
    print(f"  Salvage Value:           Rs. {motor_a['salvage_value']:,.2f}")
    print(f"  PW of Salvage:           Rs. {motor_a['pw_salvage']:,.2f}")
    print(f"  TOTAL PRESENT WORTH:     Rs. {motor_a['present_worth']:,.2f}")

    print("\n" + "-" * 80)
    print("MOTOR B - ANALYSIS")
    print("-" * 80)
    motor_b = results['Motor B']
    print(f"Initial Cost:              Rs. {motor_b['initial_cost']:,.2f}")
    print(f"Efficiency (Full Load):    {analysis.motor_b_eta_full * 100:.2f}%")
    print(f"Efficiency (Half Load):    {analysis.motor_b_eta_half * 100:.2f}%")
    print(f"\nAnnual Operating Costs:")
    print(f"  Energy Consumption:      {motor_b['annual_energy_kwh']:,.2f} kWh")
    print(f"  Energy Cost:             Rs. {motor_b['annual_energy_cost']:,.2f}")
    print(f"  Maintenance Cost:        Rs. {motor_b['annual_maintenance']:,.2f}")
    print(f"  Total Annual Cost:       Rs. {motor_b['total_annual_cost']:,.2f}")
    print(f"\nPresent Worth Analysis:")
    print(f"  PW of Operating Costs:   Rs. {motor_b['pw_operating']:,.2f}")
    print(f"  Salvage Value:           Rs. {motor_b['salvage_value']:,.2f}")
    print(f"  PW of Salvage:           Rs. {motor_b['pw_salvage']:,.2f}")
    print(f"  TOTAL PRESENT WORTH:     Rs. {motor_b['present_worth']:,.2f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nRecommended Motor: {results['recommendation']}")
    print(f"Savings over {analysis.life_years} years: Rs. {results['savings']:,.2f}")
    print()

    if results['recommendation'] == 'Motor A':
        print("Justification: Although Motor A has a higher initial cost,")
        print("its superior efficiency results in lower operating costs,")
        print("leading to lower total present worth over the motor's lifetime.")
    else:
        print("Justification: Motor B's lower initial cost and maintenance")
        print("outweigh its lower efficiency, resulting in lower total")
        print("present worth over the motor's lifetime.")

    print("\n" + "=" * 80)

    # Run dynamic simulation demo
    print("\n" + "=" * 80)
    print("DYNAMIC MOTOR SIMULATION DEMO")
    print("=" * 80)

    simulator = MotorDynamicSimulator()
    print(f"\nRunning simulation with RK45 solver for 0.5 seconds...")
    print(f"Motor Power: {simulator.rated_power/1000:.1f} kW")
    print(f"Load Torque: {simulator.load_torque} N.m")

    for i in range(500):
        speed, torque, current, power = simulator.simulate_step(method='rk45')

    print(f"\nResults at t = {simulator.time:.3f} s:")
    print(f"  Speed:   {speed:.2f} RPM")
    print(f"  Torque:  {torque:.2f} N.m")
    print(f"  Current: {current:.2f} A")
    print(f"  Power:   {power/1000:.2f} kW")

    print("\n" + "=" * 80)
