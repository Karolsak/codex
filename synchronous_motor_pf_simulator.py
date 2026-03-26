"""
3-Phase Synchronous Motor Power Factor Correction Simulator
===========================================================
Comprehensive Analysis Tool for Electrical Engineering

Problem Statement:
  A 3-phase synchronous motor takes a load of 50 kW, connected in parallel with a
  factory load of 250 kW at 0.8 lagging p.f.  The overall load p.f. must be improved
  to 0.9 lagging.  Find the leading kVAR supplied by the motor and the power factor
  at which it operates.

Tabs:
  1. Power Factor Analysis  – phasor / power-triangle, interactive sliders
  2. Fault Current          – sub-transient / transient / steady-state fault currents
  3. Protection Coordination – overcurrent relay TCC curves
  4. Speed Controller        – PID & Fuzzy step-load-change response
  5. Thermal & Economic      – temperature-rise model, life-cycle costing
  6. Harmonic & Power Quality – THD, harmonic spectrum, distortion indices
  7. Comprehensive Dashboard  – summary of all analyses
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import threading
import time
import math


# ──────────────────────────────────────────────
# CORE MATHEMATICAL MODEL
# ──────────────────────────────────────────────

class PowerFactorModel:
    """Solve the parallel-load power-factor correction problem."""

    def __init__(self, p_factory=250.0, pf_factory=0.8, p_motor=50.0, pf_required=0.9, voltage_kv=0.4):
        self.p_factory = p_factory
        self.pf_factory = pf_factory
        self.p_motor = p_motor
        self.pf_required = pf_required
        self.voltage_kv = voltage_kv

    def solve(self):
        pf = self.pf_factory
        pf_req = self.pf_required
        pf_f = min(max(pf, 0.01), 0.9999)
        pf_r = min(max(pf_req, 0.01), 0.9999)

        theta_f = math.acos(pf_f)
        Q_factory = self.p_factory * math.tan(theta_f)
        S_factory = self.p_factory / pf_f

        P_total = self.p_factory + self.p_motor
        theta_req = math.acos(pf_r)
        Q_total_required = P_total * math.tan(theta_req)
        S_total = P_total / pf_r

        Q_motor = Q_factory - Q_total_required      # leading (capacitive) kVAR
        S_motor = math.sqrt(self.p_motor**2 + Q_motor**2)
        pf_motor = self.p_motor / S_motor if S_motor > 0 else 1.0

        V_line = max(self.voltage_kv, 0.001)          # kV line-to-line
        I_factory = S_factory / (math.sqrt(3) * V_line)   # A (S in kVA, V in kV)
        I_motor = S_motor / (math.sqrt(3) * V_line)
        I_total = S_total / (math.sqrt(3) * V_line)

        return {
            "P_factory": self.p_factory,
            "pf_factory": pf_f,
            "theta_factory_deg": math.degrees(theta_f),
            "Q_factory": Q_factory,
            "S_factory": S_factory,
            "P_motor": self.p_motor,
            "P_total": P_total,
            "pf_required": pf_r,
            "theta_req_deg": math.degrees(theta_req),
            "Q_total_required": Q_total_required,
            "S_total": S_total,
            "Q_motor_leading": Q_motor,
            "S_motor": S_motor,
            "pf_motor": pf_motor,
            "pf_motor_leading": True,
            "I_factory": I_factory,
            "I_motor": I_motor,
            "I_total": I_total,
        }


class FaultCurrentModel:
    """Sub-transient, transient and steady-state fault current calculation."""

    def __init__(self, Vn=1.0, Xd=1.2, Xd_prime=0.2, Xd_pp=0.12, Td_prime=1.0, Td_pp=0.03, Ta=0.05):
        self.Vn = Vn
        self.Xd = Xd
        self.Xd_prime = Xd_prime
        self.Xd_pp = Xd_pp
        self.Td_prime = Td_prime
        self.Td_pp = Td_pp
        self.Ta = Ta

    def current(self, t):
        """Per-unit fault current envelope (peak value)."""
        Ipp = self.Vn / self.Xd_pp
        Ip = self.Vn / self.Xd_prime
        I = self.Vn / self.Xd

        ac = ((Ipp - Ip) * np.exp(-t / self.Td_pp) +
              (Ip - I) * np.exp(-t / self.Td_prime) +
              I)
        dc = Ipp * np.exp(-t / self.Ta)
        return ac * np.sqrt(2), ac, dc

    def simulate(self, t_end=0.5, dt=0.0005):
        t = np.arange(0, t_end, dt)
        peak, ac, dc = self.current(t)
        i_total = ac * np.cos(2 * np.pi * 50 * t) * np.sqrt(2) + dc
        return t, i_total, ac, dc, peak


class SpeedControllerModel:
    """Synchronous motor speed control with PID and simplified Fuzzy logic."""

    def __init__(self, J=0.5, B=0.01, Kt=1.0, rated_speed=314.16):  # 50 Hz, 2-pole → 3000 rpm
        self.J = J           # moment of inertia (kg·m²)
        self.B = B           # friction coefficient
        self.Kt = Kt         # torque constant
        self.rated_speed = rated_speed  # rad/s

    # ── PID ──────────────────────────────────────────────────────────────────
    def simulate_pid(self, Kp=5.0, Ki=2.0, Kd=0.1, load_step=0.3, t_end=5.0, dt=0.001):
        t = np.arange(0, t_end, dt)
        omega = np.zeros(len(t))
        e_int = 0.0
        e_prev = 0.0
        ref = self.rated_speed

        for i in range(1, len(t)):
            T_load = load_step * self.rated_speed if t[i] >= t_end / 2 else 0.0
            e = ref - omega[i - 1]
            e_int = np.clip(e_int + e * dt, -50, 50)
            e_der = (e - e_prev) / dt
            T_ctrl = self.Kt * (Kp * e + Ki * e_int + Kd * e_der)
            T_ctrl = np.clip(T_ctrl, 0, 3 * self.rated_speed)
            domega = (T_ctrl - self.B * omega[i - 1] - T_load) / self.J
            omega[i] = omega[i - 1] + domega * dt
            e_prev = e

        return t, omega / self.rated_speed

    # ── Fuzzy (Mamdani-like, simplified) ─────────────────────────────────────
    def _fuzzy_output(self, e, de):
        """Return normalised control output in [-1, 1]."""
        def tri(x, a, b, c):
            if x <= a or x >= c:
                return 0.0
            if x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            return (c - x) / (c - b) if c != b else 1.0

        # Membership functions for error (normalised to rated)
        NB = tri(e, -1.0, -0.6, -0.2)
        NS = tri(e, -0.5,  0.0, 0.5)   # actually small error around 0
        # simplified: treat NS as "near zero" for both neg and pos
        NS_neg = tri(e, -0.4, -0.15, 0.0)
        NS_pos = tri(e, 0.0,  0.15, 0.4)
        PB = tri(e, 0.2,   0.6, 1.0)

        # Map to crisp output via centroid (approximate)
        out = NB * (-0.8) + NS_neg * (-0.3) + NS_pos * 0.3 + PB * 0.8
        # de contribution
        out += 0.2 * np.sign(de) * min(abs(de), 0.5)
        return np.clip(out, -1.0, 1.0)

    def simulate_fuzzy(self, load_step=0.3, t_end=5.0, dt=0.001):
        t = np.arange(0, t_end, dt)
        omega = np.zeros(len(t))
        ref = self.rated_speed
        e_prev = 0.0

        for i in range(1, len(t)):
            T_load = load_step * self.rated_speed if t[i] >= t_end / 2 else 0.0
            e_norm = (ref - omega[i - 1]) / ref
            de_norm = (e_norm - e_prev / ref) / max(dt, 1e-9)
            de_norm = np.clip(de_norm, -1, 1)
            u = self._fuzzy_output(e_norm, de_norm)
            T_ctrl = max(0, u * 2 * self.rated_speed * self.Kt)
            domega = (T_ctrl - self.B * omega[i - 1] - T_load) / self.J
            omega[i] = omega[i - 1] + domega * dt
            e_prev = ref - omega[i - 1]

        return t, omega / self.rated_speed


class ThermalModel:
    """First-order motor thermal model: τ dΔT/dt = P_loss·R_th - ΔT"""

    def __init__(self, tau=600, R_th=1.0):
        self.tau = tau      # thermal time constant (s)
        self.R_th = R_th    # thermal resistance (°C/kW)

    def simulate(self, P_losses_kW, t_end=3600, dt=10, T_amb=25.0):
        t = np.arange(0, t_end, dt)
        T = np.zeros(len(t))
        T[0] = T_amb

        for i in range(1, len(t)):
            # P_losses_kW × R_th [°C/kW] = steady-state temperature rise (°C)
            dT = (P_losses_kW * self.R_th - (T[i - 1] - T_amb)) * dt / self.tau
            T[i] = T[i - 1] + dT

        return t / 3600, T   # hours, °C


class EconomicModel:
    """Life-cycle cost / payback analysis."""

    @staticmethod
    def annual_savings(P_kW, pf_old, pf_new, hours=8760, tariff=0.12):
        """Estimate annual savings from PF correction (reduced apparent-power kVA charge)."""
        S_old = P_kW / pf_old
        S_new = P_kW / pf_new
        delta_kVA = S_old - S_new
        # Assume kVA demand charge of $10/kVA/month
        monthly_demand_saving = delta_kVA * 10
        return monthly_demand_saving * 12

    @staticmethod
    def payback(capital_cost, annual_saving):
        if annual_saving <= 0:
            return float("inf")
        return capital_cost / annual_saving

    @staticmethod
    def npv(capital_cost, annual_saving, discount=0.08, years=15):
        pvf = sum(1 / (1 + discount) ** y for y in range(1, years + 1))
        return annual_saving * pvf - capital_cost


class HarmonicModel:
    """Voltage/current harmonic spectrum and power quality indices."""

    @staticmethod
    def waveform(t, fundamental=50.0, harmonics=None):
        """
        Build a distorted waveform.
        harmonics: list of (order, amplitude_fraction, phase_deg)
        """
        if harmonics is None:
            harmonics = [(1, 1.0, 0), (3, 0.05, 10), (5, 0.10, 20),
                         (7, 0.08, 30), (11, 0.04, 0), (13, 0.03, 0)]
        v = np.zeros_like(t)
        for n, amp, phi in harmonics:
            v += amp * np.sin(2 * np.pi * fundamental * n * t + np.radians(phi))
        return v

    @staticmethod
    def thd(harmonics):
        """Total Harmonic Distortion (%)."""
        if not harmonics:
            return 0.0
        fund = next((a for n, a, _ in harmonics if n == 1), 1.0)
        higher = [a for n, a, _ in harmonics if n != 1]
        thd_val = math.sqrt(sum(a**2 for a in higher)) / fund * 100
        return thd_val

    @staticmethod
    def power_factor_indices(harmonics, pf_disp=0.9):
        thd_val = HarmonicModel.thd(harmonics) / 100
        pf_dist = 1.0 / math.sqrt(1 + thd_val**2)
        return pf_disp * pf_dist, pf_dist


# ──────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────

def make_labeled_slider(parent, label, from_, to, resolution, default, row,
                        col=0, fmt="{:.2f}", colspan=1):
    """Return (frame, variable, slider) packed into parent at grid position."""
    frm = ttk.Frame(parent)
    frm.grid(row=row, column=col, columnspan=colspan, sticky="ew", padx=4, pady=2)
    ttk.Label(frm, text=label, width=26, anchor="w").pack(side="left")
    var = tk.DoubleVar(value=default)
    sl = ttk.Scale(frm, variable=var, from_=from_, to=to, orient="horizontal", length=160)
    sl.pack(side="left", padx=4)
    lbl = ttk.Label(frm, text=fmt.format(default), width=8)
    lbl.pack(side="left")

    def _update(_=None):
        lbl.config(text=fmt.format(var.get()))

    var.trace_add("write", _update)
    return frm, var, sl


def embed_figure(fig, parent):
    """Embed a matplotlib Figure into a Tkinter parent widget."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()
    return canvas


# ──────────────────────────────────────────────
# TAB 1 – POWER FACTOR ANALYSIS
# ──────────────────────────────────────────────

class PFAnalysisTab:
    def __init__(self, notebook):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="⚡ PF Analysis")
        self._build()

    def _build(self):
        pane = ttk.PanedWindow(self.tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        # ── LEFT: controls ───────────────────────────────────────────
        ctrl = ttk.Frame(pane, width=320)
        pane.add(ctrl, weight=1)

        ttk.Label(ctrl, text="Power Factor Correction Analysis",
                  font=("Helvetica", 11, "bold")).grid(row=0, column=0, columnspan=2,
                                                       sticky="w", padx=6, pady=6)
        ttk.Separator(ctrl, orient="horizontal").grid(row=1, column=0, columnspan=2,
                                                       sticky="ew", padx=4)
        ttk.Label(ctrl, text="Input Parameters", font=("Helvetica", 9, "bold italic")).grid(
            row=2, column=0, sticky="w", padx=6, pady=(6, 0))

        _, self.v_p_factory, _ = make_labeled_slider(ctrl, "Factory Load P (kW)", 10, 1000, 1, 250, 3)
        _, self.v_pf_factory, _ = make_labeled_slider(ctrl, "Factory PF (lagging)", 0.5, 0.99, 0.01, 0.80, 4)
        _, self.v_p_motor, _ = make_labeled_slider(ctrl, "Motor Load P (kW)", 1, 500, 1, 50, 5)
        _, self.v_pf_required, _ = make_labeled_slider(ctrl, "Required Overall PF", 0.5, 0.99, 0.01, 0.90, 6)

        # Voltage line for assumed system voltage
        _, self.v_voltage, _ = make_labeled_slider(ctrl, "System Voltage (kV)", 0.1, 33.0, 0.1, 0.4, 7)

        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=8, column=0, columnspan=2, sticky="ew", padx=4, pady=8)
        ttk.Button(btn_frame, text="▶ Calculate", command=self.calculate).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="↺ Reset", command=self.reset).pack(side="left", padx=4)

        # Results text
        self.result_var = tk.StringVar(value="Press 'Calculate' to see results.")
        result_lbl = ttk.Label(ctrl, textvariable=self.result_var, justify="left",
                               wraplength=300, relief="sunken", padding=6)
        result_lbl.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=6, pady=4)
        ctrl.rowconfigure(9, weight=1)
        ctrl.columnconfigure(0, weight=1)

        # ── RIGHT: matplotlib figures ────────────────────────────────
        viz = ttk.Frame(pane)
        pane.add(viz, weight=3)

        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.ax_phasor = self.fig.add_subplot(2, 2, 1)
        self.ax_tri = self.fig.add_subplot(2, 2, 2)
        self.ax_bar = self.fig.add_subplot(2, 2, 3)
        self.ax_pie = self.fig.add_subplot(2, 2, 4)

        self.canvas = embed_figure(self.fig, viz)

        # Auto-calculate with defaults on start
        self.calculate()

    def calculate(self):
        m = PowerFactorModel(
            p_factory=self.v_p_factory.get(),
            pf_factory=self.v_pf_factory.get(),
            p_motor=self.v_p_motor.get(),
            pf_required=self.v_pf_required.get(),
            voltage_kv=self.v_voltage.get(),
        )
        r = m.solve()
        self._update_results(r)
        self._draw_plots(r)

    def _update_results(self, r):
        lines = [
            "═══ FACTORY LOAD ═══",
            f"  Active Power P  = {r['P_factory']:.1f} kW",
            f"  Power Factor    = {r['pf_factory']:.3f} lag",
            f"  Angle θ         = {r['theta_factory_deg']:.2f}°",
            f"  Reactive Q      = {r['Q_factory']:.2f} kVAR (lag)",
            f"  Apparent S      = {r['S_factory']:.2f} kVA",
            "",
            "═══ COMBINED SYSTEM ═══",
            f"  Total P         = {r['P_total']:.1f} kW",
            f"  Required PF     = {r['pf_required']:.3f} lag",
            f"  Required Q      = {r['Q_total_required']:.2f} kVAR (lag)",
            f"  Required S      = {r['S_total']:.2f} kVA",
            "",
            "═══ SYNCHRONOUS MOTOR ═══",
            f"  Motor P (input) = {r['P_motor']:.1f} kW",
            f"  ★ Leading kVAR  = {r['Q_motor_leading']:.2f} kVAR (lead)",
            f"  Apparent S      = {r['S_motor']:.2f} kVA",
            f"  ★ Motor PF      = {r['pf_motor']:.4f} leading",
            f"  Motor Angle     = {math.degrees(math.acos(r['pf_motor'])):.2f}°",
        ]
        self.result_var.set("\n".join(lines))

    def _draw_plots(self, r):
        for ax in (self.ax_phasor, self.ax_tri, self.ax_bar, self.ax_pie):
            ax.clear()

        # ── 1. Phasor Diagram ─────────────────────────────────────────
        ax = self.ax_phasor
        V = 1.0
        # Factory current phasor
        pf_f = r["pf_factory"]
        th_f = -math.acos(pf_f)
        I_f = r["S_factory"] / r["P_total"]     # normalised
        Ifx, Ify = I_f * math.cos(th_f), I_f * math.sin(th_f)

        # Motor current phasor (leading: drawn above the real axis, angle positive = leading)
        pf_m = r["pf_motor"]
        th_m = math.acos(pf_m)   # magnitude of lead angle; sin(th_m) > 0 places phasor above axis
        I_m = r["S_motor"] / r["P_total"]
        Imx, Imy = I_m * math.cos(th_m), I_m * math.sin(th_m)

        # Total current phasor
        pf_t = r["pf_required"]
        th_t = -math.acos(pf_t)
        I_t = r["S_total"] / r["P_total"]
        Itx, Ity = I_t * math.cos(th_t), I_t * math.sin(th_t)

        ax.annotate("", xy=(Ifx, Ify), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.8))
        ax.annotate("", xy=(Imx, Imy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.8))
        ax.annotate("", xy=(Itx, Ity), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="green", lw=2.2))
        ax.annotate("", xy=(V, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2.0))

        ax.text(Ifx * 1.05, Ify * 1.05, f"I_factory\n(PF={pf_f:.2f}lag)", color="red", fontsize=7, ha="center")
        ax.text(Imx * 1.05, Imy * 1.05, f"I_motor\n(PF={pf_m:.3f}lead)", color="blue", fontsize=7, ha="center")
        ax.text(Itx * 1.05, Ity * 1.05, f"I_total\n(PF={pf_t:.2f}lag)", color="green", fontsize=7, ha="center")
        ax.text(V * 1.05, 0.05, "V", color="black", fontsize=8)

        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.set_aspect("equal")
        ax.set_title("Phasor Diagram", fontsize=9)
        ax.set_xlabel("Real axis"); ax.set_ylabel("Imaginary axis")

        # ── 2. Power Triangle ─────────────────────────────────────────
        ax = self.ax_tri
        Pf = r["P_factory"]
        Qf = r["Q_factory"]
        Pt = r["P_total"]
        Qt = r["Q_total_required"]
        Qm = r["Q_motor_leading"]

        ax.plot([0, Pf, Pf, 0], [0, 0, Qf, 0], "r-o", lw=1.5, ms=4, label="Factory")
        ax.plot([0, Pt, Pt, 0], [0, 0, Qt, 0], "g--o", lw=1.5, ms=4, label="Total (required)")
        ax.annotate("", xy=(Pt, Qt - Qm), xytext=(Pt, Qt),
                    arrowprops=dict(arrowstyle="<->", color="blue", lw=1.5))
        ax.text(Pt + 2, (Qt + Qt - Qm) / 2,
                f"Q_motor\n={Qm:.1f} kVAR\n(leading)", color="blue", fontsize=7)
        ax.set_title("Power Triangle", fontsize=9)
        ax.set_xlabel("Active Power P (kW)"); ax.set_ylabel("Reactive Power Q (kVAR)")
        ax.legend(fontsize=7)
        ax.grid(True, ls="--", alpha=0.5)

        # ── 3. kVA / kW / kVAR Bar Chart ─────────────────────────────
        ax = self.ax_bar
        categories = ["Factory Load", "Motor", "Combined"]
        P_vals = [r["P_factory"], r["P_motor"], r["P_total"]]
        Q_vals = [r["Q_factory"], -r["Q_motor_leading"], r["Q_total_required"]]
        S_vals = [r["S_factory"], r["S_motor"], r["S_total"]]

        x = np.arange(len(categories))
        w = 0.25
        ax.bar(x - w, P_vals, w, label="P (kW)", color="steelblue")
        ax.bar(x, Q_vals, w, label="Q (kVAR)", color="tomato")
        ax.bar(x + w, S_vals, w, label="S (kVA)", color="goldenrod")
        ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=7)
        ax.set_title("Power Comparison (kW / kVAR / kVA)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, axis="y", ls="--", alpha=0.5)

        # ── 4. PF Improvement Pie ─────────────────────────────────────
        ax = self.ax_pie
        labels = [f"Factory\nPF={r['pf_factory']:.2f}", f"Motor\nPF={r['pf_motor']:.3f}",
                  f"Combined\nPF={r['pf_required']:.2f}"]
        sizes = [r["S_factory"], r["S_motor"], r["S_total"]]
        colors = ["#FF7F7F", "#7FB3FF", "#7FFF7F"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 7})
        ax.set_title("Apparent Power Share (kVA)", fontsize=9)

        self.canvas.draw()

    def reset(self):
        self.v_p_factory.set(250.0)
        self.v_pf_factory.set(0.80)
        self.v_p_motor.set(50.0)
        self.v_pf_required.set(0.90)
        self.v_voltage.set(0.4)
        self.calculate()


# ──────────────────────────────────────────────
# TAB 2 – FAULT CURRENT
# ──────────────────────────────────────────────

class FaultCurrentTab:
    def __init__(self, notebook):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="⚠ Fault Current")
        self._running = False
        self._thread = None
        self._build()

    def _build(self):
        pane = ttk.PanedWindow(self.tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        ctrl = ttk.Frame(pane, width=280)
        pane.add(ctrl, weight=1)

        ttk.Label(ctrl, text="Fault Current Parameters",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=6)

        _, self.v_xd_pp, _ = make_labeled_slider(ctrl, "X''d sub-transient (pu)", 0.05, 0.5, 0.01, 0.12, 1)
        _, self.v_xd_p, _ = make_labeled_slider(ctrl, "X'd transient (pu)", 0.1, 1.0, 0.01, 0.20, 2)
        _, self.v_xd, _ = make_labeled_slider(ctrl, "Xd synchronous (pu)", 0.5, 3.0, 0.05, 1.20, 3)
        _, self.v_tdpp, _ = make_labeled_slider(ctrl, "T''d (s)", 0.01, 0.15, 0.005, 0.03, 4)
        _, self.v_tdp, _ = make_labeled_slider(ctrl, "T'd (s)", 0.3, 5.0, 0.1, 1.00, 5)
        _, self.v_ta, _ = make_labeled_slider(ctrl, "Ta armature (s)", 0.01, 0.3, 0.01, 0.05, 6)
        _, self.v_tend, _ = make_labeled_slider(ctrl, "Simulation time (s)", 0.1, 2.0, 0.05, 0.5, 7)

        btn = ttk.Frame(ctrl)
        btn.grid(row=8, column=0, sticky="ew", padx=4, pady=6)
        ttk.Button(btn, text="▶ Start", command=self.start).pack(side="left", padx=3)
        ttk.Button(btn, text="■ Stop", command=self.stop).pack(side="left", padx=3)
        ttk.Button(btn, text="↺ Reset", command=self.reset).pack(side="left", padx=3)

        self.info = ttk.Label(ctrl, text="", justify="left", relief="sunken", padding=4, wraplength=260)
        self.info.grid(row=9, column=0, sticky="nsew", padx=6, pady=4)
        ctrl.rowconfigure(9, weight=1)
        ctrl.columnconfigure(0, weight=1)

        viz = ttk.Frame(pane)
        pane.add(viz, weight=3)

        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3 = self.fig.add_subplot(3, 1, 3)
        self.canvas = embed_figure(self.fig, viz)

        self.simulate()

    def simulate(self):
        fm = FaultCurrentModel(
            Xd=self.v_xd.get(), Xd_prime=self.v_xd_p.get(), Xd_pp=self.v_xd_pp.get(),
            Td_prime=self.v_tdp.get(), Td_pp=self.v_tdpp.get(), Ta=self.v_ta.get()
        )
        t, i_total, ac, dc, peak = fm.simulate(t_end=self.v_tend.get())

        Ipp = 1.0 / self.v_xd_pp.get()
        Ip = 1.0 / self.v_xd_p.get()
        I = 1.0 / self.v_xd.get()
        self.info.config(text=(
            f"Sub-transient peak : {Ipp * math.sqrt(2):.3f} pu\n"
            f"Transient peak     : {Ip * math.sqrt(2):.3f} pu\n"
            f"Steady-state peak  : {I * math.sqrt(2):.3f} pu\n"
            f"DC offset (t=0)    : {Ipp:.3f} pu"
        ))

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.clear()

        self.ax1.plot(t * 1000, i_total, "b", lw=0.8, label="Total fault current")
        self.ax1.plot(t * 1000, peak, "r--", lw=1.2, label="AC envelope (peak)")
        self.ax1.set_title("Fault Current Waveform (pu)", fontsize=9)
        self.ax1.set_xlabel("Time (ms)"); self.ax1.set_ylabel("Current (pu)")
        self.ax1.legend(fontsize=7); self.ax1.grid(True, ls="--", alpha=0.4)

        self.ax2.plot(t * 1000, ac, "r", lw=1.2, label="AC RMS envelope")
        self.ax2.plot(t * 1000, dc, "k--", lw=1.0, label="DC component")
        self.ax2.set_title("AC & DC Components (pu)", fontsize=9)
        self.ax2.set_xlabel("Time (ms)"); self.ax2.set_ylabel("Current (pu)")
        self.ax2.legend(fontsize=7); self.ax2.grid(True, ls="--", alpha=0.4)

        self.ax3.semilogy(t * 1000, ac, "r", lw=1.2)
        self.ax3.set_title("AC Envelope (log scale)", fontsize=9)
        self.ax3.set_xlabel("Time (ms)"); self.ax3.set_ylabel("Current (pu) – log")
        self.ax3.grid(True, ls="--", alpha=0.4, which="both")

        self.canvas.draw()

    def start(self):
        self.simulate()

    def stop(self):
        self._running = False

    def reset(self):
        self.v_xd_pp.set(0.12); self.v_xd_p.set(0.20); self.v_xd.set(1.20)
        self.v_tdpp.set(0.03); self.v_tdp.set(1.00); self.v_ta.set(0.05)
        self.v_tend.set(0.5)
        self.simulate()


# ──────────────────────────────────────────────
# TAB 3 – PROTECTION COORDINATION
# ──────────────────────────────────────────────

class ProtectionTab:
    def __init__(self, notebook):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="🛡 Protection")
        self._build()

    def _build(self):
        pane = ttk.PanedWindow(self.tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        ctrl = ttk.Frame(pane, width=300)
        pane.add(ctrl, weight=1)

        ttk.Label(ctrl, text="Overcurrent Relay Settings",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=6)

        _, self.v_rated_I, _ = make_labeled_slider(ctrl, "Rated Current (A)", 1, 2000, 1, 400, 1)
        _, self.v_ct_ratio, _ = make_labeled_slider(ctrl, "CT Ratio (n:1)", 50, 4000, 50, 400, 2)
        _, self.v_relay1_ps, _ = make_labeled_slider(ctrl, "Relay 1 Plug Setting (A)", 0.5, 2.0, 0.05, 1.0, 3)
        _, self.v_relay1_tms, _ = make_labeled_slider(ctrl, "Relay 1 TMS", 0.05, 1.5, 0.05, 0.2, 4)
        _, self.v_relay2_ps, _ = make_labeled_slider(ctrl, "Relay 2 Plug Setting (A)", 0.5, 2.0, 0.05, 1.25, 5)
        _, self.v_relay2_tms, _ = make_labeled_slider(ctrl, "Relay 2 TMS", 0.05, 1.5, 0.05, 0.5, 6)
        _, self.v_if_pu, _ = make_labeled_slider(ctrl, "Fault Current (pu)", 1, 20, 0.5, 10.0, 7)

        btn = ttk.Frame(ctrl)
        btn.grid(row=8, column=0, sticky="ew", padx=4, pady=6)
        ttk.Button(btn, text="▶ Calculate", command=self.calculate).pack(side="left", padx=3)
        ttk.Button(btn, text="↺ Reset", command=self.reset).pack(side="left", padx=3)

        self.info = ttk.Label(ctrl, text="", justify="left", relief="sunken", padding=4, wraplength=270)
        self.info.grid(row=9, column=0, sticky="nsew", padx=6, pady=4)
        ctrl.rowconfigure(9, weight=1); ctrl.columnconfigure(0, weight=1)

        viz = ttk.Frame(pane)
        pane.add(viz, weight=3)
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.ax_tcc = self.fig.add_subplot(1, 2, 1)
        self.ax_coord = self.fig.add_subplot(1, 2, 2)
        self.canvas = embed_figure(self.fig, viz)

        self.calculate()

    @staticmethod
    def _iec_normal_inverse(I_mult, tms):
        """IEC Standard Inverse operating time (s)."""
        I_mult = np.maximum(I_mult, 1.001)
        return tms * 0.14 / (I_mult**0.02 - 1)

    @staticmethod
    def _iec_very_inverse(I_mult, tms):
        """IEC Very Inverse operating time (s)."""
        I_mult = np.maximum(I_mult, 1.001)
        return tms * 13.5 / (I_mult - 1)

    def calculate(self):
        I_r = self.v_rated_I.get()
        ct = self.v_ct_ratio.get()
        ps1 = self.v_relay1_ps.get()
        tms1 = self.v_relay1_tms.get()
        ps2 = self.v_relay2_ps.get()
        tms2 = self.v_relay2_tms.get()
        If_pu = self.v_if_pu.get()

        I_fault = If_pu * I_r
        I_sec = I_fault / ct
        Im1 = I_sec / ps1
        Im2 = I_sec / ps2
        t1 = float(self._iec_very_inverse(np.array([Im1]), tms1)[0])
        t2 = float(self._iec_normal_inverse(np.array([Im2]), tms2)[0])

        self.info.config(text=(
            f"Fault current        : {I_fault:.1f} A\n"
            f"CT secondary current : {I_sec:.3f} A\n"
            f"Relay 1 mult. (Im/Ps): {Im1:.2f}×\n"
            f"Relay 1 trip time    : {t1:.3f} s  (Very Inverse)\n"
            f"Relay 2 mult. (Im/Ps): {Im2:.2f}×\n"
            f"Relay 2 trip time    : {t2:.3f} s  (Normal Inverse)\n"
            f"Coordination margin  : {abs(t2 - t1):.3f} s"
        ))

        M = np.linspace(1.05, 20, 500)
        t_vi1 = self._iec_very_inverse(M, tms1)
        t_ni2 = self._iec_normal_inverse(M, tms2)
        I_sec_vals = M * ps1 * ct   # primary current for relay 1

        ax = self.ax_tcc; ax.clear()
        ax.loglog(I_sec_vals, t_vi1, "r-", lw=2, label=f"Relay 1 (VI, TMS={tms1})")
        ax.loglog(M * ps2 * ct, t_ni2, "b--", lw=2, label=f"Relay 2 (NI, TMS={tms2})")
        ax.axvline(I_fault, color="k", ls=":", lw=1.5, label=f"If={I_fault:.0f}A")
        ax.scatter([I_fault], [t1], color="red", zorder=5, s=60)
        ax.scatter([I_fault], [t2], color="blue", zorder=5, s=60)
        ax.set_title("Time-Current Characteristics", fontsize=9)
        ax.set_xlabel("Primary Current (A)"); ax.set_ylabel("Operating Time (s)")
        ax.legend(fontsize=7); ax.grid(True, which="both", ls="--", alpha=0.4)

        ax2 = self.ax_coord; ax2.clear()
        ax2.bar(["Relay 1\n(Very Inv)", "Relay 2\n(Norm Inv)"], [t1, t2],
                color=["red", "blue"])
        ax2.axhline(0.3, ls="--", color="k", lw=1, label="0.3 s CTI minimum")
        ax2.set_title(f"Trip Times at {If_pu:.1f}×If", fontsize=9)
        ax2.set_ylabel("Time (s)")
        ax2.legend(fontsize=7); ax2.grid(True, axis="y", ls="--", alpha=0.4)

        self.canvas.draw()

    def reset(self):
        self.v_rated_I.set(400); self.v_ct_ratio.set(400)
        self.v_relay1_ps.set(1.0); self.v_relay1_tms.set(0.2)
        self.v_relay2_ps.set(1.25); self.v_relay2_tms.set(0.5)
        self.v_if_pu.set(10.0)
        self.calculate()


# ──────────────────────────────────────────────
# TAB 4 – SPEED CONTROLLER
# ──────────────────────────────────────────────

class SpeedControllerTab:
    def __init__(self, notebook):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="🔄 Speed Ctrl")
        self._running = False
        self._thread = None
        self._build()

    def _build(self):
        pane = ttk.PanedWindow(self.tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        ctrl = ttk.Frame(pane, width=300)
        pane.add(ctrl, weight=1)

        ttk.Label(ctrl, text="Speed Controller Parameters",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=6)

        _, self.v_J, _ = make_labeled_slider(ctrl, "Inertia J (kg·m²)", 0.05, 5.0, 0.05, 0.5, 1)
        _, self.v_B, _ = make_labeled_slider(ctrl, "Friction B", 0.001, 0.1, 0.001, 0.01, 2)
        _, self.v_Kp, _ = make_labeled_slider(ctrl, "PID Kp", 0.5, 20.0, 0.5, 5.0, 3)
        _, self.v_Ki, _ = make_labeled_slider(ctrl, "PID Ki", 0.0, 10.0, 0.1, 2.0, 4)
        _, self.v_Kd, _ = make_labeled_slider(ctrl, "PID Kd", 0.0, 2.0, 0.05, 0.1, 5)
        _, self.v_step, _ = make_labeled_slider(ctrl, "Load Step (pu)", 0.0, 1.0, 0.05, 0.3, 6)
        _, self.v_tend, _ = make_labeled_slider(ctrl, "Sim Time (s)", 1, 20, 0.5, 5.0, 7)

        btn = ttk.Frame(ctrl)
        btn.grid(row=8, column=0, sticky="ew", padx=4, pady=6)
        ttk.Button(btn, text="▶ Start", command=self.start).pack(side="left", padx=3)
        ttk.Button(btn, text="■ Stop", command=self.stop).pack(side="left", padx=3)
        ttk.Button(btn, text="↺ Reset", command=self.reset).pack(side="left", padx=3)

        self.info = ttk.Label(ctrl, text="", justify="left", relief="sunken", padding=4, wraplength=270)
        self.info.grid(row=9, column=0, sticky="nsew", padx=6, pady=4)
        ctrl.rowconfigure(9, weight=1); ctrl.columnconfigure(0, weight=1)

        viz = ttk.Frame(pane)
        pane.add(viz, weight=3)
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.canvas = embed_figure(self.fig, viz)

        self.simulate()

    def simulate(self):
        sc = SpeedControllerModel(J=self.v_J.get(), B=self.v_B.get())
        t_end = self.v_tend.get()
        step = self.v_step.get()
        Kp, Ki, Kd = self.v_Kp.get(), self.v_Ki.get(), self.v_Kd.get()

        t_pid, w_pid = sc.simulate_pid(Kp=Kp, Ki=Ki, Kd=Kd, load_step=step, t_end=t_end)
        t_fuz, w_fuz = sc.simulate_fuzzy(load_step=step, t_end=t_end)

        # Performance metrics
        t_half = t_end / 2
        idx_step = np.searchsorted(t_pid, t_half)
        w_ss_pid = np.mean(w_pid[-50:]) if len(w_pid) > 50 else w_pid[-1]
        w_ss_fuz = np.mean(w_fuz[-50:]) if len(w_fuz) > 50 else w_fuz[-1]
        overshoot_pid = max(0, np.max(w_pid) - 1) * 100
        overshoot_fuz = max(0, np.max(w_fuz) - 1) * 100

        self.info.config(text=(
            f"PID  → SS speed: {w_ss_pid:.4f} pu, overshoot: {overshoot_pid:.2f}%\n"
            f"Fuzzy→ SS speed: {w_ss_fuz:.4f} pu, overshoot: {overshoot_fuz:.2f}%\n"
            f"Load step at t={t_half:.1f}s: +{step:.2f} pu"
        ))

        for ax in (self.ax1, self.ax2):
            ax.clear()

        self.ax1.plot(t_pid, w_pid, "b-", lw=1.5, label="PID")
        self.ax1.plot(t_fuz, w_fuz, "r--", lw=1.5, label="Fuzzy")
        self.ax1.axhline(1.0, color="k", ls=":", lw=1, label="Reference")
        self.ax1.axvline(t_end / 2, color="gray", ls="--", lw=1, label="Load step")
        self.ax1.set_title("Speed Response (normalised pu)", fontsize=9)
        self.ax1.set_xlabel("Time (s)"); self.ax1.set_ylabel("ω / ω_rated")
        self.ax1.legend(fontsize=7); self.ax1.grid(True, ls="--", alpha=0.4)
        self.ax1.set_ylim(0, 1.2)

        # Error signal
        e_pid = 1.0 - w_pid
        e_fuz = 1.0 - w_fuz
        self.ax2.plot(t_pid, e_pid, "b-", lw=1.2, label="PID error")
        self.ax2.plot(t_fuz, e_fuz, "r--", lw=1.2, label="Fuzzy error")
        self.ax2.axhline(0, color="k", ls="-", lw=0.8)
        self.ax2.axvline(t_end / 2, color="gray", ls="--", lw=1)
        self.ax2.set_title("Speed Error (pu)", fontsize=9)
        self.ax2.set_xlabel("Time (s)"); self.ax2.set_ylabel("Error (pu)")
        self.ax2.legend(fontsize=7); self.ax2.grid(True, ls="--", alpha=0.4)

        self.canvas.draw()

    def start(self):
        self._running = True
        self.simulate()

    def stop(self):
        self._running = False

    def reset(self):
        self.v_J.set(0.5); self.v_B.set(0.01)
        self.v_Kp.set(5.0); self.v_Ki.set(2.0); self.v_Kd.set(0.1)
        self.v_step.set(0.3); self.v_tend.set(5.0)
        self.simulate()


# ──────────────────────────────────────────────
# TAB 5 – THERMAL & ECONOMIC ANALYSIS
# ──────────────────────────────────────────────

class ThermalEconomicTab:
    def __init__(self, notebook):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="🌡 Thermal/Economic")
        self._build()

    def _build(self):
        pane = ttk.PanedWindow(self.tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        ctrl = ttk.Frame(pane, width=300)
        pane.add(ctrl, weight=1)

        ttk.Label(ctrl, text="Thermal & Economic Parameters",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=6)

        _, self.v_P_loss, _ = make_labeled_slider(ctrl, "Motor Losses (kW)", 0.5, 50.0, 0.5, 5.0, 1)
        _, self.v_tau, _ = make_labeled_slider(ctrl, "Thermal Tau (min)", 5, 120, 5, 60, 2, fmt="{:.0f}")
        _, self.v_Rth, _ = make_labeled_slider(ctrl, "Thermal Resis (°C/kW)", 0.1, 5.0, 0.1, 1.0, 3)
        _, self.v_Tamb, _ = make_labeled_slider(ctrl, "Ambient Temp (°C)", -10, 60, 1, 25, 4, fmt="{:.0f}")
        _, self.v_t_run, _ = make_labeled_slider(ctrl, "Run time (hours)", 1, 72, 1, 12, 5, fmt="{:.0f}")

        ttk.Separator(ctrl).grid(row=6, column=0, sticky="ew", padx=4, pady=4)
        ttk.Label(ctrl, text="Economic Parameters", font=("Helvetica", 9, "bold italic")).grid(
            row=7, column=0, sticky="w", padx=6)

        _, self.v_P_load, _ = make_labeled_slider(ctrl, "Load P (kW)", 10, 1000, 10, 300, 8)
        _, self.v_pf_old, _ = make_labeled_slider(ctrl, "Old PF", 0.5, 0.99, 0.01, 0.80, 9)
        _, self.v_pf_new, _ = make_labeled_slider(ctrl, "New PF", 0.5, 0.99, 0.01, 0.90, 10)
        _, self.v_capital, _ = make_labeled_slider(ctrl, "Capital Cost ($)", 100, 50000, 100, 5000, 11, fmt="{:.0f}")
        _, self.v_discount, _ = make_labeled_slider(ctrl, "Discount Rate (%)", 1, 20, 0.5, 8.0, 12)

        btn = ttk.Frame(ctrl)
        btn.grid(row=13, column=0, sticky="ew", padx=4, pady=6)
        ttk.Button(btn, text="▶ Analyse", command=self.calculate).pack(side="left", padx=3)
        ttk.Button(btn, text="↺ Reset", command=self.reset).pack(side="left", padx=3)

        self.info = ttk.Label(ctrl, text="", justify="left", relief="sunken", padding=4, wraplength=270)
        self.info.grid(row=14, column=0, sticky="nsew", padx=6, pady=4)
        ctrl.rowconfigure(14, weight=1); ctrl.columnconfigure(0, weight=1)

        viz = ttk.Frame(pane)
        pane.add(viz, weight=3)
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.ax_temp = self.fig.add_subplot(2, 2, 1)
        self.ax_econ = self.fig.add_subplot(2, 2, 2)
        self.ax_npv = self.fig.add_subplot(2, 2, 3)
        self.ax_payback = self.fig.add_subplot(2, 2, 4)
        self.canvas = embed_figure(self.fig, viz)

        self.calculate()

    def calculate(self):
        # Thermal
        tau_s = self.v_tau.get() * 60
        Rth = self.v_Rth.get()
        t_run_s = self.v_t_run.get() * 3600
        tm = ThermalModel(tau=tau_s, R_th=Rth)
        t_hrs, T = tm.simulate(P_losses_kW=self.v_P_loss.get(), t_end=t_run_s,
                               T_amb=self.v_Tamb.get())
        T_max = float(np.max(T))
        T_ss = self.v_Tamb.get() + self.v_P_loss.get() * Rth

        # Economic
        P = self.v_P_load.get(); pf_o = self.v_pf_old.get(); pf_n = self.v_pf_new.get()
        capital = self.v_capital.get()
        disc = self.v_discount.get() / 100.0
        ann_sav = EconomicModel.annual_savings(P, pf_o, pf_n)
        pb = EconomicModel.payback(capital, ann_sav)
        npv_val = EconomicModel.npv(capital, ann_sav, discount=disc)

        discount_rates = np.linspace(0.02, 0.20, 50)
        npv_curve = [EconomicModel.npv(capital, ann_sav, d) for d in discount_rates]

        years = np.arange(1, 16)
        cum_savings = ann_sav * years - capital

        self.info.config(text=(
            f"Thermal Steady-State : {T_ss:.1f} °C\n"
            f"Peak Temperature     : {T_max:.1f} °C\n\n"
            f"Annual PF Savings    : ${ann_sav:,.0f}\n"
            f"Simple Payback       : {pb:.1f} years\n"
            f"NPV (15 yr, {disc*100:.0f}%)  : ${npv_val:,.0f}\n"
            f"{'PROFITABLE ✔' if npv_val > 0 else 'NOT PROFITABLE ✗'}"
        ))

        for ax in (self.ax_temp, self.ax_econ, self.ax_npv, self.ax_payback):
            ax.clear()

        self.ax_temp.plot(t_hrs, T, "r-", lw=1.5)
        self.ax_temp.axhline(T_ss, ls="--", color="k", lw=1, label=f"T_ss={T_ss:.1f}°C")
        self.ax_temp.axhline(130, ls=":", color="red", lw=1, label="Class B limit 130°C")
        self.ax_temp.set_title("Motor Temperature Rise", fontsize=9)
        self.ax_temp.set_xlabel("Time (hours)"); self.ax_temp.set_ylabel("Temperature (°C)")
        self.ax_temp.legend(fontsize=7); self.ax_temp.grid(True, ls="--", alpha=0.4)

        self.ax_econ.bar(["Old PF kVA", "New PF kVA", "Reduction"],
                         [P / pf_o, P / pf_n, P / pf_o - P / pf_n],
                         color=["salmon", "skyblue", "green"])
        self.ax_econ.set_title("Apparent Power Comparison (kVA)", fontsize=9)
        self.ax_econ.set_ylabel("kVA"); self.ax_econ.grid(True, axis="y", ls="--", alpha=0.4)

        self.ax_npv.plot(discount_rates * 100, npv_curve, "b-", lw=1.5)
        self.ax_npv.axhline(0, color="k", ls="-", lw=0.8)
        self.ax_npv.fill_between(discount_rates * 100, npv_curve, 0,
                                  where=[v > 0 for v in npv_curve], alpha=0.2, color="green", label="Profitable")
        self.ax_npv.fill_between(discount_rates * 100, npv_curve, 0,
                                  where=[v <= 0 for v in npv_curve], alpha=0.2, color="red", label="Loss")
        self.ax_npv.set_title("NPV vs Discount Rate (15 yr)", fontsize=9)
        self.ax_npv.set_xlabel("Discount Rate (%)"); self.ax_npv.set_ylabel("NPV ($)")
        self.ax_npv.legend(fontsize=7); self.ax_npv.grid(True, ls="--", alpha=0.4)

        self.ax_payback.plot(years, cum_savings, "g-", lw=1.5, marker="o", ms=4)
        self.ax_payback.axhline(0, color="k", ls="-", lw=0.8)
        self.ax_payback.fill_between(years, cum_savings, 0,
                                      where=[v > 0 for v in cum_savings], alpha=0.2, color="green")
        self.ax_payback.fill_between(years, cum_savings, 0,
                                      where=[v <= 0 for v in cum_savings], alpha=0.2, color="red")
        self.ax_payback.set_title("Cumulative Savings Over Time", fontsize=9)
        self.ax_payback.set_xlabel("Year"); self.ax_payback.set_ylabel("Cumulative Savings ($)")
        self.ax_payback.grid(True, ls="--", alpha=0.4)

        self.canvas.draw()

    def reset(self):
        self.v_P_loss.set(5.0); self.v_tau.set(60); self.v_Rth.set(1.0)
        self.v_Tamb.set(25); self.v_t_run.set(12)
        self.v_P_load.set(300); self.v_pf_old.set(0.80); self.v_pf_new.set(0.90)
        self.v_capital.set(5000); self.v_discount.set(8.0)
        self.calculate()


# ──────────────────────────────────────────────
# TAB 6 – HARMONIC & POWER QUALITY
# ──────────────────────────────────────────────

class HarmonicTab:
    def __init__(self, notebook):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="〜 Harmonics")
        self._build()

    def _build(self):
        pane = ttk.PanedWindow(self.tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        ctrl = ttk.Frame(pane, width=300)
        pane.add(ctrl, weight=1)

        ttk.Label(ctrl, text="Harmonic & Power Quality",
                  font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Label(ctrl, text="Harmonic amplitudes (% of fundamental):",
                  font=("Helvetica", 8, "italic")).grid(row=1, column=0, sticky="w", padx=6)

        _, self.v_h3, _ = make_labeled_slider(ctrl, "3rd Harmonic (%)", 0, 30, 0.5, 5.0, 2, fmt="{:.1f}")
        _, self.v_h5, _ = make_labeled_slider(ctrl, "5th Harmonic (%)", 0, 30, 0.5, 10.0, 3, fmt="{:.1f}")
        _, self.v_h7, _ = make_labeled_slider(ctrl, "7th Harmonic (%)", 0, 25, 0.5, 8.0, 4, fmt="{:.1f}")
        _, self.v_h11, _ = make_labeled_slider(ctrl, "11th Harmonic (%)", 0, 20, 0.5, 4.0, 5, fmt="{:.1f}")
        _, self.v_h13, _ = make_labeled_slider(ctrl, "13th Harmonic (%)", 0, 20, 0.5, 3.0, 6, fmt="{:.1f}")
        _, self.v_pf_disp, _ = make_labeled_slider(ctrl, "Displacement PF", 0.5, 1.0, 0.01, 0.90, 7)
        _, self.v_fund_freq, _ = make_labeled_slider(ctrl, "Fund. Frequency (Hz)", 45, 65, 0.5, 50.0, 8, fmt="{:.1f}")

        btn = ttk.Frame(ctrl)
        btn.grid(row=9, column=0, sticky="ew", padx=4, pady=6)
        ttk.Button(btn, text="▶ Calculate", command=self.calculate).pack(side="left", padx=3)
        ttk.Button(btn, text="↺ Reset", command=self.reset).pack(side="left", padx=3)

        self.info = ttk.Label(ctrl, text="", justify="left", relief="sunken", padding=4, wraplength=270)
        self.info.grid(row=10, column=0, sticky="nsew", padx=6, pady=4)
        ctrl.rowconfigure(10, weight=1); ctrl.columnconfigure(0, weight=1)

        viz = ttk.Frame(pane)
        pane.add(viz, weight=3)
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.ax_wave = self.fig.add_subplot(2, 2, 1)
        self.ax_spec = self.fig.add_subplot(2, 2, 2)
        self.ax_pq = self.fig.add_subplot(2, 2, 3)
        self.ax_iec = self.fig.add_subplot(2, 2, 4)
        self.canvas = embed_figure(self.fig, viz)

        self.calculate()

    def calculate(self):
        h3 = self.v_h3.get() / 100
        h5 = self.v_h5.get() / 100
        h7 = self.v_h7.get() / 100
        h11 = self.v_h11.get() / 100
        h13 = self.v_h13.get() / 100
        pf_d = self.v_pf_disp.get()
        f = self.v_fund_freq.get()

        harmonics = [(1, 1.0, 0), (3, h3, 10), (5, h5, 20),
                     (7, h7, 30), (11, h11, 0), (13, h13, 0)]

        thd_val = HarmonicModel.thd(harmonics)
        pf_total, pf_dist = HarmonicModel.power_factor_indices(harmonics, pf_d)

        t = np.linspace(0, 2 / f, 1000)
        v_dist = HarmonicModel.waveform(t, fundamental=f, harmonics=harmonics)
        v_pure = np.sin(2 * np.pi * f * t)

        # IEC 61000-3-2 Class A limits (simplified)
        iec_limits = {3: 2.30, 5: 1.14, 7: 0.77, 9: 0.40, 11: 0.33, 13: 0.21}
        amp_pu = {3: h3, 5: h5, 7: h7, 11: h11, 13: h13}

        self.info.config(text=(
            f"THD (voltage)        : {thd_val:.2f} %\n"
            f"Displacement PF      : {pf_d:.3f}\n"
            f"Distortion Factor    : {pf_dist:.4f}\n"
            f"True Power Factor    : {pf_total:.4f}\n"
            f"  (True PF = Disp PF × Dist. Factor)\n\n"
            f"IEC 61000 THD limit  : 8 % (Class A)\n"
            f"Status: {'PASS ✔' if thd_val < 8 else 'FAIL ✗'}"
        ))

        for ax in (self.ax_wave, self.ax_spec, self.ax_pq, self.ax_iec):
            ax.clear()

        # Waveform
        self.ax_wave.plot(t * 1000, v_dist, "b-", lw=1.0, label="Distorted")
        self.ax_wave.plot(t * 1000, v_pure, "k--", lw=0.8, alpha=0.5, label="Pure sine")
        self.ax_wave.set_title(f"Voltage Waveform (THD={thd_val:.1f}%)", fontsize=9)
        self.ax_wave.set_xlabel("Time (ms)"); self.ax_wave.set_ylabel("Voltage (pu)")
        self.ax_wave.legend(fontsize=7); self.ax_wave.grid(True, ls="--", alpha=0.4)

        # Harmonic spectrum
        orders = [1, 3, 5, 7, 11, 13]
        amps = [1.0, h3, h5, h7, h11, h13]
        colors = ["steelblue" if n == 1 else "tomato" for n in orders]
        self.ax_spec.bar([str(n) for n in orders], [a * 100 for a in amps], color=colors)
        self.ax_spec.set_title("Harmonic Spectrum (% of fundamental)", fontsize=9)
        self.ax_spec.set_xlabel("Harmonic Order"); self.ax_spec.set_ylabel("Amplitude (%)")
        self.ax_spec.grid(True, axis="y", ls="--", alpha=0.4)

        # Power quality indices bar
        pq_names = ["Disp. PF", "Dist. Factor", "True PF"]
        pq_vals = [pf_d, pf_dist, pf_total]
        colors_pq = ["steelblue", "goldenrod", "green"]
        bars = self.ax_pq.bar(pq_names, pq_vals, color=colors_pq)
        self.ax_pq.set_ylim(0, 1.1)
        for bar, val in zip(bars, pq_vals):
            self.ax_pq.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                             f"{val:.3f}", ha="center", fontsize=8)
        self.ax_pq.set_title("Power Quality Indices", fontsize=9)
        self.ax_pq.set_ylabel("Index (pu)"); self.ax_pq.grid(True, axis="y", ls="--", alpha=0.4)

        # IEC compliance
        harm_orders = sorted(iec_limits.keys())
        actual_vals = [amp_pu.get(h, 0) * 100 for h in harm_orders]
        limit_vals = [iec_limits[h] for h in harm_orders]
        x_pos = np.arange(len(harm_orders))
        self.ax_iec.bar(x_pos - 0.2, actual_vals, 0.35,
                        label="Actual", color=["green" if a <= l else "red"
                                               for a, l in zip(actual_vals, limit_vals)])
        self.ax_iec.bar(x_pos + 0.2, limit_vals, 0.35, label="IEC Limit", color="gray", alpha=0.5)
        self.ax_iec.set_xticks(x_pos)
        self.ax_iec.set_xticklabels([f"{h}th" for h in harm_orders])
        self.ax_iec.set_title("IEC 61000-3-2 Compliance (Class A)", fontsize=9)
        self.ax_iec.set_ylabel("Amplitude (A rms)")
        self.ax_iec.legend(fontsize=7); self.ax_iec.grid(True, axis="y", ls="--", alpha=0.4)

        self.canvas.draw()

    def reset(self):
        self.v_h3.set(5.0); self.v_h5.set(10.0); self.v_h7.set(8.0)
        self.v_h11.set(4.0); self.v_h13.set(3.0)
        self.v_pf_disp.set(0.90); self.v_fund_freq.set(50.0)
        self.calculate()


# ──────────────────────────────────────────────
# TAB 7 – COMPREHENSIVE DASHBOARD
# ──────────────────────────────────────────────

class DashboardTab:
    def __init__(self, notebook):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="📊 Dashboard")
        self._build()

    def _build(self):
        # Top: parameter summary
        top = ttk.LabelFrame(self.tab, text="Problem Statement", padding=6)
        top.pack(fill="x", padx=6, pady=4)

        problem_text = (
            "A 3-phase synchronous motor takes a load of 50 kW connected in parallel with "
            "a factory load of 250 kW at 0.8 lagging p.f.  The overall load p.f. must be "
            "improved to 0.9 lagging.  Find: (1) Leading kVAR from the motor, "
            "(2) Power factor at which the motor operates."
        )
        ttk.Label(top, text=problem_text, wraplength=900, justify="left",
                  font=("Helvetica", 9)).pack(anchor="w")

        # Middle: step-by-step solution
        sol_frame = ttk.LabelFrame(self.tab, text="Step-by-Step Solution", padding=6)
        sol_frame.pack(fill="x", padx=6, pady=4)

        r = PowerFactorModel().solve()
        steps = self._build_steps(r)
        ttk.Label(sol_frame, text=steps, justify="left", font=("Courier", 8),
                  wraplength=900).pack(anchor="w")

        # Bottom: comprehensive charts
        viz_frame = ttk.Frame(self.tab)
        viz_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self.fig = Figure(figsize=(10, 5), tight_layout=True)
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        self.canvas = embed_figure(self.fig, viz_frame)

        self._draw(r)

    @staticmethod
    def _build_steps(r):
        lines = [
            "STEP 1 – Factory load reactive power",
            f"  θ_factory = arccos({r['pf_factory']:.2f}) = {r['theta_factory_deg']:.2f}°",
            f"  Q_factory = {r['P_factory']:.0f} × tan({r['theta_factory_deg']:.2f}°) = {r['Q_factory']:.2f} kVAR (lagging)",
            "",
            "STEP 2 – Total active power",
            f"  P_total = {r['P_factory']:.0f} + {r['P_motor']:.0f} = {r['P_total']:.0f} kW",
            "",
            "STEP 3 – Required reactive power at PF = 0.9 lag",
            f"  θ_req = arccos({r['pf_required']:.2f}) = {r['theta_req_deg']:.2f}°",
            f"  Q_total = {r['P_total']:.0f} × tan({r['theta_req_deg']:.2f}°) = {r['Q_total_required']:.2f} kVAR",
            "",
            "STEP 4 – Leading kVAR supplied by the motor",
            f"  Q_motor = Q_factory − Q_total = {r['Q_factory']:.2f} − {r['Q_total_required']:.2f} = {r['Q_motor_leading']:.2f} kVAR (leading)",
            "",
            "STEP 5 – Motor apparent power and power factor",
            f"  S_motor = √(P²+Q²) = √({r['P_motor']:.0f}² + {r['Q_motor_leading']:.2f}²) = {r['S_motor']:.2f} kVA",
            f"  PF_motor = {r['P_motor']:.0f} / {r['S_motor']:.2f} = {r['pf_motor']:.4f} LEADING",
            "",
            f"  ★  ANSWER 1:  Leading kVAR = {r['Q_motor_leading']:.2f} kVAR",
            f"  ★  ANSWER 2:  Motor PF     = {r['pf_motor']:.4f} leading  ({math.degrees(math.acos(r['pf_motor'])):.2f}° lead)",
        ]
        return "\n".join(lines)

    def _draw(self, r):
        # Sankey-like power flow
        ax = self.ax1; ax.clear()
        ax.set_xlim(0, 4); ax.set_ylim(-1, 3)
        ax.add_patch(plt.Rectangle((0.1, 1.5), 0.8, 0.7, color="steelblue", alpha=0.7))
        ax.text(0.5, 1.85, f"Factory\n{r['P_factory']:.0f}kW\n{r['Q_factory']:.0f}kVAR lag",
                ha="center", va="center", fontsize=7, color="white")
        ax.add_patch(plt.Rectangle((0.1, 0.4), 0.8, 0.7, color="tomato", alpha=0.7))
        ax.text(0.5, 0.75, f"Motor\n{r['P_motor']:.0f}kW\n{r['Q_motor_leading']:.1f}kVAR lead",
                ha="center", va="center", fontsize=7, color="white")
        ax.add_patch(plt.Rectangle((2.8, 1.1), 0.9, 0.8, color="green", alpha=0.7))
        ax.text(3.25, 1.5, f"Combined\n{r['P_total']:.0f}kW\n{r['Q_total_required']:.1f}kVAR lag",
                ha="center", va="center", fontsize=7, color="white")
        ax.annotate("", xy=(2.8, 1.5), xytext=(0.9, 1.85),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5))
        ax.annotate("", xy=(2.8, 1.5), xytext=(0.9, 0.75),
                    arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5))
        ax.set_title("Power Flow Summary", fontsize=9)
        ax.axis("off")

        # PF comparison gauge-like bar
        ax2 = self.ax2; ax2.clear()
        pf_values = [r["pf_factory"], r["pf_required"], r["pf_motor"]]
        labels = [f"Factory\n{r['pf_factory']:.3f} lag",
                  f"Combined\n{r['pf_required']:.3f} lag",
                  f"Motor\n{r['pf_motor']:.4f} lead"]
        colors = ["tomato", "green", "steelblue"]
        bars = ax2.barh(labels, pf_values, color=colors, height=0.5)
        ax2.set_xlim(0, 1.1)
        for bar, val in zip(bars, pf_values):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)
        ax2.axvline(1.0, color="k", ls="--", lw=1)
        ax2.set_title("Power Factor Comparison", fontsize=9)
        ax2.set_xlabel("Power Factor"); ax2.grid(True, axis="x", ls="--", alpha=0.4)

        # Reactive power waterfall
        ax3 = self.ax3; ax3.clear()
        categories = ["Factory Q (lag)", "−Motor Q (lead)", "Net Q (lag)"]
        values = [r["Q_factory"], -r["Q_motor_leading"], r["Q_total_required"]]
        colors_wf = ["red" if v > 0 else "blue" for v in values]
        bars = ax3.bar(categories, values, color=colors_wf, alpha=0.7)
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() / 2 if val > 0 else bar.get_height() - 8,
                     f"{val:.1f}", ha="center", fontsize=8, color="white", fontweight="bold")
        ax3.set_title("Reactive Power Balance (kVAR)", fontsize=9)
        ax3.set_ylabel("kVAR"); ax3.grid(True, axis="y", ls="--", alpha=0.4)
        ax3.tick_params(axis="x", labelsize=7)

        self.canvas.draw()


# ──────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────

class SynchronousMotorApp(tk.Tk):
    """Main application window with notebook tabs."""

    def __init__(self):
        super().__init__()
        self.title("3-Phase Synchronous Motor – Power Factor & Comprehensive Analysis")
        self.geometry("1200x780")
        self.minsize(900, 600)

        self._create_menu()
        self._create_status_bar()
        self._create_notebook()

        self.bind("<Configure>", self._on_resize)

    def _create_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Reset All", command=self._reset_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

    def _create_status_bar(self):
        self.status_var = tk.StringVar(value="Ready  |  3-Phase Synchronous Motor Simulator")
        bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=(4, 1))
        bar.pack(side="bottom", fill="x")

    def _create_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self.tab_pf = PFAnalysisTab(self.notebook)
        self.tab_fault = FaultCurrentTab(self.notebook)
        self.tab_prot = ProtectionTab(self.notebook)
        self.tab_speed = SpeedControllerTab(self.notebook)
        self.tab_thermo = ThermalEconomicTab(self.notebook)
        self.tab_harm = HarmonicTab(self.notebook)
        self.tab_dash = DashboardTab(self.notebook)

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _on_tab_change(self, _event):
        tab_idx = self.notebook.index(self.notebook.select())
        names = ["PF Analysis", "Fault Current", "Protection",
                 "Speed Controller", "Thermal/Economic", "Harmonics", "Dashboard"]
        if tab_idx < len(names):
            self.status_var.set(f"Active tab: {names[tab_idx]}  |  3-Phase Synchronous Motor Simulator")

    def _on_resize(self, _event):
        """Redraw canvas on window resize for auto-scaling."""
        pass  # Canvas widgets automatically resize with pack fill=BOTH expand=True

    def _reset_all(self):
        self.tab_pf.reset()
        self.tab_fault.reset()
        self.tab_prot.reset()
        self.tab_speed.reset()
        self.tab_thermo.reset()
        self.tab_harm.reset()
        messagebox.showinfo("Reset", "All tabs have been reset to default values.")

    def _about(self):
        messagebox.showinfo(
            "About",
            "3-Phase Synchronous Motor Power Factor Correction Simulator\n\n"
            "Comprehensive electrical engineering analysis tool covering:\n"
            " • Power factor correction (PF analysis)\n"
            " • Fault current modelling (sub-transient/transient)\n"
            " • Protection relay coordination (IEC curves)\n"
            " • Speed control (PID + Fuzzy Logic)\n"
            " • Thermal & economic analysis\n"
            " • Harmonic & power quality (IEC 61000)\n"
            " • Comprehensive dashboard\n\n"
            "Default problem:\n"
            "  Factory: 250 kW @ 0.8 PF lag\n"
            "  Motor:   50 kW  (synchronous)\n"
            "  Target:  0.9 PF overall lag\n\n"
            "Results:\n"
            "  Motor leading kVAR ≈ 42.2 kVAR\n"
            "  Motor PF           ≈ 0.7643 leading"
        )


def main():
    app = SynchronousMotorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
