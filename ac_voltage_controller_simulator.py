#!/usr/bin/env python3
"""
3-Phase Delta-Connected AC Voltage Controller — Induction Motor Speed Control
==============================================================================
Comprehensive simulation, calculation and analysis tool.

Problem Statement
-----------------
A 3φ delta-connected AC voltage controller is used to control the speed of a
3φ, 5 hp, 208 V, 60 Hz induction motor.
At full-load output (5 hp, 208 V) the power factor is 0.85 lagging and the
efficiency is 90 %.

(a) Draw the circuit.
(b) Determine the input kVA for the full-load output condition.
(c) What is the range of the firing angle for the full-load condition?
(d) Determine the current and voltage ratings of the thyristors.
(e) If the power factor is 0.80 (lagging), for a firing angle of 60°, draw
    qualitatively the waveform of motor current and voltage for one phase.

Tabs
----
 1  Overview & Circuit Diagram
 2  Motor Parameters (sliders, live update)
 3  Calculations — answers a–e with detailed explanation
 4  Waveforms — voltage / current
 5  Dynamic Simulation — start / stop / reset
 6  Fault Current Analysis
 7  Protection Coordination (TCC curves)
 8  Speed Controller — PID & Fuzzy with step load
 9  Thermal & Economic Analysis
10  Harmonic & Power Quality
"""

import sys
import threading
import time
import warnings
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
HP2W    = 746.0
TWO_PI  = 2.0 * np.pi
SQRT2   = np.sqrt(2.0)
SQRT3   = np.sqrt(3.0)


# ===========================================================================
# CALCULATION ENGINE
# ===========================================================================

def motor_params(hp, V_line, freq, pf, eta, poles):
    """Return dict with all derived electrical quantities."""
    P_out   = hp * HP2W
    P_in    = P_out / eta
    S_in    = P_in  / pf
    Q_in    = S_in  * np.sqrt(max(0.0, 1.0 - pf**2))
    phi     = np.arccos(np.clip(pf, 1e-6, 1.0))
    phi_deg = np.degrees(phi)

    # 3-phase delta-connected load
    I_line  = S_in / (SQRT3 * V_line)
    I_phase = I_line / SQRT3          # delta: I_phase = I_line / √3

    Vm      = SQRT2 * V_line
    V_thy   = Vm                      # peak inverse voltage
    I_thy_pk = SQRT2 * I_phase        # peak thyristor current
    I_thy_rm = I_phase / SQRT2        # rms per thyristor (half-cycle conduction)
    I_thy_av = I_thy_pk / np.pi       # average per thyristor

    n_sync  = 120.0 * freq / poles
    w_sync  = TWO_PI * n_sync / 60.0

    # Full-load torque (assume 5 % rated slip)
    s_rated = 0.05
    w_r     = w_sync * (1.0 - s_rated)
    T_fl    = P_out / w_r if w_r > 0 else 0.0

    return dict(
        P_out=P_out, P_in=P_in, S_in=S_in, Q_in=Q_in,
        phi_deg=phi_deg, phi_rad=phi,
        I_line=I_line, I_phase=I_phase, Vm=Vm,
        V_thy=V_thy, I_thy_pk=I_thy_pk, I_thy_rm=I_thy_rm, I_thy_av=I_thy_av,
        alpha_min=phi_deg, alpha_max=180.0,
        n_sync=n_sync, w_sync=w_sync, T_full=T_fl,
    )


def controller_waveform(V_line, freq, alpha_deg, pf_load, cycles=3):
    """
    Single-phase equivalent waveforms for AC voltage controller with RL load.
    Returns t [s], v_source [V], v_output [V], i_output [A normalised].
    """
    alpha = np.radians(alpha_deg)
    phi   = np.arccos(np.clip(pf_load, 1e-9, 1.0))
    omega = TWO_PI * freq
    Vm    = SQRT2 * V_line
    Z     = 1.0   # normalised impedance — gives shape

    N  = 6000
    t  = np.linspace(0.0, cycles / freq, N)
    wt = omega * t

    v_s   = Vm * np.sin(wt)
    v_out = np.zeros(N)
    i_out = np.zeros(N)

    for k in range(cycles * 2):
        fire = k * np.pi + alpha                # firing angle in wt domain
        ext  = k * np.pi + np.pi + phi          # approximate extinction angle

        if fire >= wt[-1]:
            break

        idx = np.where((wt >= fire) & (wt < ext))[0]
        if len(idx) == 0:
            continue

        v_out[idx] = Vm * np.sin(wt[idx])
        theta = wt[idx] - fire
        sign  = 1 if k % 2 == 0 else -1
        with np.errstate(over="ignore", invalid="ignore"):
            i_val = sign * (Vm / Z) * (
                np.sin(wt[idx] - phi) -
                np.sin(fire - phi) * np.exp(-theta / np.tan(phi + 1e-9))
            )
        i_out[idx] = np.where(np.isfinite(i_val), i_val, 0.0)

    return t, v_s, v_out, i_out


def fft_analysis(signal, dt):
    """Compute one-sided FFT.  Returns freqs [Hz], amplitudes, THD [%]."""
    N     = len(signal)
    freqs = np.fft.rfftfreq(N, d=dt)
    mag   = np.abs(np.fft.rfft(signal)) * 2.0 / N

    fundamental = 60.0
    tol = 3.0
    h1m = np.abs(freqs - fundamental) < tol
    V1  = mag[h1m].max() if h1m.any() else 1e-9

    harm_sq = 0.0
    for h in range(2, 21):
        hm = np.abs(freqs - h * fundamental) < tol
        harm_sq += (mag[hm].max() if hm.any() else 0.0) ** 2

    thd = 100.0 * np.sqrt(harm_sq) / (V1 + 1e-12)
    return freqs, mag, thd


# ---------------------------------------------------------------------------
# Fault current  (3-phase symmetrical short-circuit at motor terminals)
# ---------------------------------------------------------------------------

def fault_current(V_line, freq, Z_source_pu, S_base_kva):
    """
    Return time arrays and phase currents during a bolted 3-phase fault.
    Z_source_pu is total source impedance on S_base_kva base.
    """
    V_base = V_line / SQRT3          # phase voltage
    Z_base = V_line**2 / (S_base_kva * 1e3)
    Z_ohm  = Z_source_pu * Z_base

    # Symmetrical fault rms current
    I_sc_rms  = V_base / max(Z_ohm, 1e-9)
    I_sc_peak = SQRT2 * I_sc_rms * (1.0 + np.exp(-1.0 / 0.1))  # with DC offset

    omega = TWO_PI * freq
    t = np.linspace(0.0, 0.2, 2000)

    # Phase A fault current with DC offset (R/X = 0.1 → tau = X/(w*R))
    tau = 1.0 / (omega * 0.1)        # decay time constant
    i_a = SQRT2 * I_sc_rms * (
        np.sin(omega * t - np.pi / 2) +
        np.sin(np.pi / 2) * np.exp(-t / tau)
    )
    i_b = SQRT2 * I_sc_rms * (
        np.sin(omega * t - np.pi / 2 - TWO_PI / 3) +
        np.sin(np.pi / 2 + TWO_PI / 3) * np.exp(-t / tau)
    )
    i_c = SQRT2 * I_sc_rms * (
        np.sin(omega * t - np.pi / 2 + TWO_PI / 3) +
        np.sin(np.pi / 2 - TWO_PI / 3) * np.exp(-t / tau)
    )
    return t, i_a, i_b, i_c, I_sc_rms, I_sc_peak


# ---------------------------------------------------------------------------
# Protection TCC (IEC standard inverse)
# ---------------------------------------------------------------------------

def tcc_curve(I_arr, Ip, TMS, curve='SI'):
    """
    IEC time-overcurrent curve.
    curve: 'SI' standard inverse, 'VI' very inverse, 'EI' extremely inverse
    """
    params = {
        'SI': (0.14,  0.02),
        'VI': (13.5,  1.0),
        'EI': (80.0,  2.0),
    }
    k, n = params.get(curve, (0.14, 0.02))
    ratio = np.clip(I_arr / Ip, 1.001, 1e6)
    t = TMS * k / (ratio**n - 1.0)
    return np.clip(t, 0.0, 100.0)


# ===========================================================================
# MOTOR DYNAMICS MODEL
# ===========================================================================

class MotorModel:
    """
    Simplified equivalent-circuit induction motor model.
    State: rotor angular speed ω_r [rad/s].
    Input: effective stator voltage V_rms, load torque T_L.
    Integration: Euler with small fixed dt.
    """
    # Nameplate: 5 hp, 208 V, 60 Hz, 4-pole (typical parameter set)
    R1 = 0.641;  R2 = 0.332
    X1 = 1.106;  X2 = 0.464
    Xm = 26.3
    J  = 0.05    # kg·m²  moment of inertia
    Bm = 0.005   # N·m·s  viscous friction

    def __init__(self, poles=4, freq=60.0, V_rated=208.0):
        self.poles   = poles
        self.freq    = freq
        self.V_rated = V_rated
        self.reset()

    def w_sync(self):
        return TWO_PI * self.freq * 2.0 / self.poles

    def reset(self):
        self.wr = 0.0
        self.t  = 0.0
        self.hist = dict(t=[], rpm=[], Te=[], I=[], Tl=[], alpha=[])

    def _torque(self, V, s):
        s   = max(s, 0.001)
        Z   = complex(self.R1 + self.R2 / s, self.X1 + self.X2)
        I2  = V / abs(Z)
        ws  = self.w_sync()
        return 3.0 * I2**2 * self.R2 / (s * max(ws, 1e-9))

    def v_effective(self, V_nom, alpha_deg):
        """RMS output voltage of single-phase AC voltage controller (approx)."""
        a = np.radians(np.clip(alpha_deg, 0.0, 180.0))
        factor = np.sqrt(np.clip(
            (1.0 / np.pi) * (2.0 * (np.pi - a) + np.sin(2.0 * a)),
            0.0, 1.0
        ))
        return V_nom * factor

    def step(self, V_nom, alpha_deg, Tl, dt=0.002):
        ws = self.w_sync()
        V  = self.v_effective(V_nom, alpha_deg)
        s  = (ws - self.wr) / ws if ws > 0.0 else 1.0
        s  = np.clip(s, 0.001, 1.0)

        Te  = self._torque(V, s)
        Tm  = Tl + self.Bm * self.wr
        dw  = (Te - Tm) / self.J

        self.wr += dw * dt
        self.wr  = np.clip(self.wr, 0.0, ws * 1.05)
        self.t  += dt

        R2s = self.R2 / max(s, 0.001)
        I   = V / max(abs(complex(self.R1 + R2s, self.X1 + self.X2)), 1e-9)

        rpm = self.wr * 60.0 / TWO_PI
        self.hist['t'].append(self.t)
        self.hist['rpm'].append(rpm)
        self.hist['Te'].append(Te)
        self.hist['I'].append(I)
        self.hist['Tl'].append(Tl)
        self.hist['alpha'].append(alpha_deg)
        return Te, I


# ===========================================================================
# PID CONTROLLER
# ===========================================================================

class PIDController:
    """
    Discrete-time PID with anti-windup clamping.

    Output units: same as (error × gain), typically RPM × gain.
    The caller is responsible for mapping the output to the plant input (e.g.
    firing angle α) via an appropriate scale factor and sign convention.
    """

    def __init__(self, kp=2.0, ki=0.5, kd=0.1, dt=0.002,
                 out_min=-1000.0, out_max=1000.0):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.dt = dt
        self.out_min = out_min; self.out_max = out_max
        self.reset()

    def reset(self):
        self._integral  = 0.0
        self._prev_err  = 0.0

    def __call__(self, sp, meas):
        err = sp - meas
        self._integral += err * self.dt
        # Anti-windup: clamp integral so ki*integral alone cannot exceed output range
        if self.ki > 1e-9:
            max_int = (self.out_max - self.out_min) / self.ki
            self._integral = float(np.clip(self._integral, -max_int, max_int))
        deriv = (err - self._prev_err) / (self.dt + 1e-9)
        self._prev_err = err
        u = self.kp * err + self.ki * self._integral + self.kd * deriv
        return float(np.clip(u, self.out_min, self.out_max))


# ===========================================================================
# FUZZY CONTROLLER
# ===========================================================================

class FuzzyController:
    """
    5-term symmetric Mamdani fuzzy speed controller.
    Inputs : speed error (e), change of error (Δe)
    Output : firing-angle adjustment Δα [degrees]
    """
    LABELS  = ['NB', 'NS', 'ZE', 'PS', 'PB']
    CENTERS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    #  Rule table [e-row][Δe-col] → output label index (output = Δα rate)
    #  e = sp - meas:  NB ↔ speed >> ref (error very negative) → increase α
    #                  PB ↔ speed << ref (error very positive)  → decrease α
    #         NB  NS  ZE  PS  PB
    RULES = [
        [0,  0,  1,  2,  3],   # e = NB: speed >> ref → large +α correction
        [0,  1,  2,  3,  4],   # e = NS
        [1,  2,  2,  2,  3],   # e = ZE
        [1,  2,  3,  4,  4],   # e = PS
        [2,  2,  3,  4,  4],   # e = PB: speed << ref → large −α correction
    ]

    def __init__(self, e_range=1000.0, de_range=200.0, out_range=80.0):
        self.e_range   = e_range
        self.de_range  = de_range
        self.out_range = out_range
        self._prev_err = 0.0

    @staticmethod
    def _trimf(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        return (x - a) / (b - a + 1e-9) if x <= b else (c - x) / (c - b + 1e-9)

    def _fuzzify(self, val):
        return [self._trimf(val, c - 0.5, c, c + 0.5) for c in self.CENTERS]

    def __call__(self, sp, meas, dt=0.002):
        err  = sp - meas
        derr = (err - self._prev_err) / (dt + 1e-9)
        self._prev_err = err

        e_n  = float(np.clip(err  / self.e_range,  -1.0, 1.0))
        de_n = float(np.clip(derr / self.de_range, -1.0, 1.0))

        mfe  = self._fuzzify(e_n)
        mfde = self._fuzzify(de_n)

        out_agg = [0.0] * 5
        for i, me in enumerate(mfe):
            for j, md in enumerate(mfde):
                strength = min(me, md)
                rule_out = self.RULES[i][j]
                out_agg[rule_out] = max(out_agg[rule_out], strength)

        num = sum(out_agg[k] * c for k, c in enumerate(self.CENTERS))
        den = sum(out_agg) + 1e-12
        return float(np.clip(num / den * self.out_range, -self.out_range, self.out_range))


# ===========================================================================
# THERMAL MODEL
# ===========================================================================

class ThermalModel:
    """First-order lumped thermal model: C·dT/dt = P_loss − (T−T_amb)/R_th."""

    def __init__(self, R_th=1.5, C_th=500.0, T_amb=25.0):
        self.R_th  = R_th    # K/W
        self.C_th  = C_th    # J/K
        self.T_amb = T_amb   # °C
        self.T     = T_amb   # current temperature

    def reset(self):
        self.T = self.T_amb

    def step(self, P_loss, dt=1.0):
        dT = (P_loss - (self.T - self.T_amb) / self.R_th) / self.C_th
        self.T += dT * dt
        return self.T

    def steady_state(self, P_loss):
        return self.T_amb + P_loss * self.R_th


# ===========================================================================
# HELPER — embed matplotlib figure in a tk frame
# ===========================================================================

def embed_figure(fig, parent, toolbar=False):
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    w = canvas.get_tk_widget()
    w.pack(fill="both", expand=True)
    if toolbar:
        tb = NavigationToolbar2Tk(canvas, parent)
        tb.update()
    return canvas


def make_fig(rows=1, cols=1, figsize=(10, 5)):
    fig = Figure(figsize=figsize, tight_layout=True)
    axes = []
    for i in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, i + 1))
    return fig, axes


# ===========================================================================
# MAIN APPLICATION
# ===========================================================================

class App(tk.Tk):
    TITLE = "3φ Delta AC Voltage Controller — Induction Motor Analysis"

    def __init__(self):
        super().__init__()
        self.title(self.TITLE)
        self.minsize(960, 640)
        try:
            self.state("zoomed")
        except Exception:
            self.geometry("1400x900")

        self._sim_running  = False
        self._ctrl_running = False
        self._sim_thread   = None
        self._ctrl_thread  = None
        self._motor        = MotorModel()
        self._thermal      = ThermalModel()

        self._build_styles()
        self._init_vars()
        self._build_notebook()
        self._update_all()
        self.bind("<Configure>", self._on_resize)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------------
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook.Tab", padding=[12, 5], font=("Helvetica", 9, "bold"))
        s.configure("Header.TLabel", font=("Helvetica", 11, "bold"), foreground="#1a237e")
        s.configure("Result.TLabel", font=("Courier", 10), background="#f5f5f5")
        s.configure("Accent.TButton", font=("Helvetica", 9, "bold"))

    # -----------------------------------------------------------------------
    def _init_vars(self):
        self.v = {
            # Motor nameplate
            "hp":        tk.DoubleVar(value=5.0),
            "V_line":    tk.DoubleVar(value=208.0),
            "freq":      tk.DoubleVar(value=60.0),
            "pf":        tk.DoubleVar(value=0.85),
            "eta":       tk.DoubleVar(value=0.90),
            "poles":     tk.IntVar(value=4),
            # Waveform
            "alpha":     tk.DoubleVar(value=31.8),
            "pf_load":   tk.DoubleVar(value=0.85),
            # Simulation
            "T_load":    tk.DoubleVar(value=15.0),
            "alpha_sim": tk.DoubleVar(value=40.0),
            # Speed controller
            "speed_ref": tk.DoubleVar(value=1710.0),
            "kp":        tk.DoubleVar(value=2.0),
            "ki":        tk.DoubleVar(value=0.5),
            "kd":        tk.DoubleVar(value=0.1),
            "step_load": tk.DoubleVar(value=20.0),
            # Fault
            "Z_src_pu":  tk.DoubleVar(value=0.05),
            # Protection
            "Ip1":       tk.DoubleVar(value=15.0),
            "tms1":      tk.DoubleVar(value=0.5),
            "Ip2":       tk.DoubleVar(value=8.0),
            "tms2":      tk.DoubleVar(value=0.2),
            # Thermal
            "R_th":      tk.DoubleVar(value=1.5),
            "hours":     tk.DoubleVar(value=8000.0),
            "energy_cost": tk.DoubleVar(value=0.10),
        }

    # -----------------------------------------------------------------------
    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=4, pady=4)

        tabs = [
            ("Overview",        self._tab_overview),
            ("Parameters",      self._tab_parameters),
            ("Calculations",    self._tab_calculations),
            ("Waveforms",       self._tab_waveforms),
            ("Simulation",      self._tab_simulation),
            ("Fault Current",   self._tab_fault),
            ("Protection",      self._tab_protection),
            ("Speed Ctrl",      self._tab_speed_ctrl),
            ("Thermal/Econ",    self._tab_thermal_econ),
            ("Harmonics",       self._tab_harmonics),
        ]
        for title, builder in tabs:
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            builder(frame)

    # =======================================================================
    # TAB 1 — OVERVIEW & CIRCUIT DIAGRAM
    # =======================================================================
    def _tab_overview(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="both", expand=True)

        # ── Circuit diagram (matplotlib canvas) ──────────────────────────
        fig = Figure(figsize=(10, 5), tight_layout=True)
        ax  = fig.add_subplot(111)
        self._draw_circuit(ax)
        self._canvas_overview = embed_figure(fig, top)

        # ── Problem text ─────────────────────────────────────────────────
        bot = ttk.LabelFrame(parent, text="Problem Statement & Theory")
        bot.pack(fill="x", padx=6, pady=4)

        txt = scrolledtext.ScrolledText(bot, height=9, wrap="word",
                                        font=("Helvetica", 9))
        txt.pack(fill="x", padx=4, pady=2)
        txt.insert("end", PROBLEM_TEXT)
        txt.config(state="disabled")

    def _draw_circuit(self, ax):
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_facecolor("#fafafa")
        ax.set_title("3φ Delta-Connected AC Voltage Controller — Circuit Diagram",
                     fontsize=11, fontweight="bold", pad=8)

        kw = dict(transform=ax.transData, color="black", lw=1.5)

        # ── 3-phase supply lines  (left side) ────────────────────────────
        colors = ["#c62828", "#2e7d32", "#1565c0"]
        y_ph   = [8, 5, 2]
        labels = ["Phase A", "Phase B", "Phase C"]
        for y, c, lbl in zip(y_ph, colors, labels):
            ax.plot([0, 2], [y, y], color=c, lw=2.5)
            ax.text(-0.1, y, lbl, ha="right", va="center", color=c,
                    fontsize=8, fontweight="bold")

        # ── Thyristor pairs (anti-parallel) in each line ─────────────────
        for idx, (y, c) in enumerate(zip(y_ph, colors)):
            # Forward thyristor symbol (triangle + bar)
            self._thyristor(ax, 3.0 + idx * 0.05, y, c, forward=True)
            # Connect to delta
            ax.plot([4.5, 6], [y, y], color=c, lw=2.0)

        # ── Delta connection box ──────────────────────────────────────────
        delta_x = [6, 10, 10, 6, 6]
        delta_y = [8,  8,  2, 2, 8]
        ax.plot(delta_x, delta_y, color="#37474f", lw=2.5, linestyle="--")
        ax.text(8, 9.2, "Δ-connected\nAC Voltage\nController",
                ha="center", va="bottom", fontsize=8, color="#37474f",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#37474f"))

        # Phase impedances across delta sides
        ax.text(10.5, 6.5, "Z_A\n(R+jωL)", ha="left", fontsize=8, color=colors[0])
        ax.text(10.5, 4.0, "Z_B\n(R+jωL)", ha="left", fontsize=8, color=colors[1])
        ax.text(8,    1.0, "Z_C  (R+jωL)", ha="center", fontsize=8, color=colors[2])

        # Draw impedance lines on delta sides
        ax.plot([10, 12], [8, 8], color=colors[0], lw=2)
        ax.plot([10, 12], [2, 2], color=colors[2], lw=2)
        ax.plot([10, 10], [8, 2], color=colors[1], lw=2)

        # Motor block
        motor_rect = mpatches.FancyBboxPatch(
            (12.2, 3.5), 2.8, 3.0,
            boxstyle="round,pad=0.2",
            linewidth=2, edgecolor="#1565c0", facecolor="#e3f2fd"
        )
        ax.add_patch(motor_rect)
        ax.text(13.6, 5.8, "3φ IM", ha="center", fontsize=10,
                fontweight="bold", color="#1565c0")
        ax.text(13.6, 5.2, "5 hp / 208 V", ha="center", fontsize=7.5,
                color="#1565c0")
        ax.text(13.6, 4.7, "60 Hz / Δ-conn.", ha="center", fontsize=7.5,
                color="#1565c0")
        ax.text(13.6, 4.2, "PF=0.85  η=90%", ha="center", fontsize=7.5,
                color="#1565c0")

        # Connect delta to motor
        for y in [8, 5, 2]:
            ax.plot([12, 12.2], [y, 5.0 + (y - 5) * 0.2], color="#37474f",
                    lw=1.5, ls=":")

        # Gate signals
        for idx, y in enumerate(y_ph):
            ax.annotate("", xy=(3.5 + idx * 0.05, y + 0.5),
                        xytext=(3.5 + idx * 0.05, y + 1.5),
                        arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.2))
            ax.text(3.5 + idx * 0.05, y + 1.7,
                    "Gate\nsignal", ha="center", fontsize=7, color="#e65100")

        ax.text(8, -0.2, "Thyristor pairs fire at angle α ∈ [φ_load , 180°] "
                "to vary RMS voltage across motor phases",
                ha="center", fontsize=8, style="italic", color="#555")

    @staticmethod
    def _thyristor(ax, x, y, color, forward=True):
        """Draw an anti-parallel thyristor pair symbol."""
        # Forward: triangle pointing right
        tri = mpatches.Polygon(
            [[x, y + 0.4], [x + 1.2, y], [x, y - 0.4]],
            closed=True, color=color, alpha=0.7
        )
        ax.add_patch(tri)
        ax.plot([x + 1.2, x + 1.2], [y - 0.4, y + 0.4], color=color, lw=2)
        # Reverse thyristor (small)
        tri2 = mpatches.Polygon(
            [[x + 1.2, y + 0.25], [x, y + 0.0], [x + 1.2, y - 0.25]],
            closed=True, color=color, alpha=0.3
        )
        ax.add_patch(tri2)
        ax.plot([x, x], [y - 0.25, y + 0.25], color=color, lw=1.5)
        ax.plot([x, x + 1.2], [y, y], color=color, lw=1.5)

    # =======================================================================
    # TAB 2 — PARAMETERS
    # =======================================================================
    def _tab_parameters(self, parent):
        lf = ttk.LabelFrame(parent, text="Motor & Controller Input Parameters")
        lf.pack(fill="both", expand=True, padx=10, pady=8)

        specs = [
            ("hp",      "Rated Power (hp)",           1.0,  20.0,  0.5),
            ("V_line",  "Line-to-Line Voltage (V)",  100.0, 600.0, 5.0),
            ("freq",    "Supply Frequency (Hz)",       50.0,  60.0, 0.5),
            ("pf",      "Full-Load Power Factor",       0.5,   1.0, 0.01),
            ("eta",     "Efficiency (η)",               0.5,   1.0, 0.01),
            ("alpha",   "Firing Angle α (°)",           0.0, 175.0, 1.0),
            ("pf_load", "Load PF (waveform tab)",       0.5,   1.0, 0.01),
            ("T_load",  "Load Torque (N·m)",            0.0,  50.0, 0.5),
            ("alpha_sim","Sim. Firing Angle (°)",       20.0, 175.0, 1.0),
        ]

        self._param_labels = {}
        for row, (key, label, lo, hi, res) in enumerate(specs):
            ttk.Label(lf, text=label, width=32).grid(
                row=row, column=0, sticky="w", padx=8, pady=3)
            sl = ttk.Scale(lf, from_=lo, to=hi, variable=self.v[key],
                           orient="horizontal", length=350,
                           command=lambda _e, k=key: self._on_slider(k))
            sl.grid(row=row, column=1, padx=6, pady=3)
            val_lbl = ttk.Label(lf, text="", width=10, style="Result.TLabel")
            val_lbl.grid(row=row, column=2, padx=4)
            self._param_labels[key] = val_lbl

        # Poles combobox
        ttk.Label(lf, text="Number of Poles", width=32).grid(
            row=len(specs), column=0, sticky="w", padx=8, pady=3)
        cb = ttk.Combobox(lf, textvariable=self.v["poles"],
                          values=[2, 4, 6, 8], width=6, state="readonly")
        cb.grid(row=len(specs), column=1, sticky="w", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._update_all())

        # Refresh button
        ttk.Button(lf, text="Refresh All Calculations",
                   style="Accent.TButton",
                   command=self._update_all).grid(
            row=len(specs) + 1, column=0, columnspan=3, pady=10)

        self._update_param_labels()

    def _on_slider(self, key):
        self._update_param_labels()
        self._update_all()

    def _update_param_labels(self):
        fmt = {
            "hp": "{:.1f}", "V_line": "{:.0f}", "freq": "{:.1f}",
            "pf": "{:.3f}", "eta": "{:.3f}", "alpha": "{:.1f}°",
            "pf_load": "{:.3f}", "T_load": "{:.1f}", "alpha_sim": "{:.1f}°",
        }
        for key, lbl in self._param_labels.items():
            f = fmt.get(key, "{:.3f}")
            try:
                val = self.v[key].get()
                lbl.config(text=f.format(val))
            except Exception:
                pass

    # =======================================================================
    # TAB 3 — CALCULATIONS
    # =======================================================================
    def _tab_calculations(self, parent):
        self._calc_text = scrolledtext.ScrolledText(
            parent, wrap="word", font=("Courier", 9), bg="#f9f9f9"
        )
        self._calc_text.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Button(parent, text="Recalculate",
                   style="Accent.TButton",
                   command=self._refresh_calc).pack(pady=4)

    def _refresh_calc(self):
        hp  = self.v["hp"].get()
        V   = self.v["V_line"].get()
        f   = self.v["freq"].get()
        pf  = self.v["pf"].get()
        eta = self.v["eta"].get()
        pol = self.v["poles"].get()
        r   = motor_params(hp, V, f, pf, eta, pol)
        self._calc_text.config(state="normal")
        self._calc_text.delete("1.0", "end")
        self._calc_text.insert("end", format_results(hp, V, f, pf, eta, pol, r))
        self._calc_text.config(state="disabled")

    # =======================================================================
    # TAB 4 — WAVEFORMS
    # =======================================================================
    def _tab_waveforms(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", padx=8, pady=4)

        for key, lbl, lo, hi in [
            ("alpha",   "Firing Angle α (°)", 0.0, 175.0),
            ("pf_load", "Load PF",             0.5,   1.0),
        ]:
            ttk.Label(ctrl, text=lbl, width=20).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=self.v[key],
                      orient="horizontal", length=220,
                      command=lambda _e: self._plot_waveform()).pack(side="left")
            ttk.Label(ctrl, textvariable=self.v[key], width=7).pack(side="left")

        self._fig_wave, axes = make_fig(2, 1, figsize=(11, 5))
        self._ax_wv, self._ax_wi = axes
        self._canvas_wave = embed_figure(self._fig_wave, parent, toolbar=True)

    def _plot_waveform(self):
        alpha   = self.v["alpha"].get()
        pf_load = self.v["pf_load"].get()
        V       = self.v["V_line"].get()
        f       = self.v["freq"].get()

        t, v_s, v_out, i_out = controller_waveform(V, f, alpha, pf_load)

        ax1, ax2 = self._ax_wv, self._ax_wi
        ax1.cla(); ax2.cla()

        ax1.plot(t * 1e3, v_s,   color="#1565c0", lw=1.2, label="Source v(t)")
        ax1.plot(t * 1e3, v_out, color="#c62828", lw=1.5, label="Output v_out(t)")
        ax1.axhline(0, color="grey", lw=0.5)
        ax1.set_ylabel("Voltage (V)")
        ax1.set_title(
            f"AC Voltage Controller — α = {alpha:.1f}°, PF = {pf_load:.2f}",
            fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t * 1e3, i_out, color="#2e7d32", lw=1.5, label="Phase current i(t)")
        ax2.axhline(0, color="grey", lw=0.5)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Current (pu)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        self._fig_wave.tight_layout()
        self._canvas_wave.draw()

    # =======================================================================
    # TAB 5 — DYNAMIC SIMULATION
    # =======================================================================
    def _tab_simulation(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Simulation Controls")
        ctrl.pack(fill="x", padx=8, pady=4)

        for key, lbl, lo, hi in [
            ("alpha_sim", "Firing Angle α (°)", 20.0, 175.0),
            ("T_load",    "Load Torque (N·m)",   0.0,  50.0),
        ]:
            ttk.Label(ctrl, text=lbl, width=22).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=self.v[key],
                      orient="horizontal", length=200).pack(side="left", padx=2)
            ttk.Label(ctrl, textvariable=self.v[key], width=6).pack(side="left")

        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(side="right", padx=8)
        ttk.Button(btn_frame, text="▶ Start", style="Accent.TButton",
                   command=self._sim_start).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="■ Stop",
                   command=self._sim_stop).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="↺ Reset",
                   command=self._sim_reset).pack(side="left", padx=3)

        self._fig_sim, axes = make_fig(3, 1, figsize=(11, 7))
        self._ax_spd, self._ax_torq, self._ax_cur = axes
        for ax, ylbl in zip(axes, ["Speed (RPM)", "Torque (N·m)", "Current (A)"]):
            ax.set_ylabel(ylbl)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        self._fig_sim.tight_layout()
        self._canvas_sim = embed_figure(self._fig_sim, parent, toolbar=False)

        self._sim_status = tk.StringVar(value="Status: Idle")
        ttk.Label(parent, textvariable=self._sim_status).pack(pady=2)

    def _sim_start(self):
        if self._sim_running:
            return
        self._sim_running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

    def _sim_stop(self):
        self._sim_running = False

    def _sim_reset(self):
        self._sim_running = False
        time.sleep(0.05)
        self._motor.reset()
        self._thermal.reset()
        for ax in [self._ax_spd, self._ax_torq, self._ax_cur]:
            ax.cla()
            ax.grid(True, alpha=0.3)
        self._ax_spd.set_ylabel("Speed (RPM)")
        self._ax_torq.set_ylabel("Torque (N·m)")
        self._ax_cur.set_ylabel("Current (A)")
        self._ax_cur.set_xlabel("Time (s)")
        self._canvas_sim.draw()
        self._sim_status.set("Status: Reset")

    def _sim_loop(self):
        self._motor.reset()
        while self._sim_running:
            alpha = self.v["alpha_sim"].get()
            Tl    = self.v["T_load"].get()
            V     = self.v["V_line"].get()
            self._motor.step(V, alpha, Tl, dt=0.004)
            if len(self._motor.hist["t"]) % 20 == 0:
                self.after(0, self._sim_update_plot)
            time.sleep(0.004)
        self.after(0, lambda: self._sim_status.set("Status: Stopped"))

    def _sim_update_plot(self):
        h = self._motor.hist
        if len(h["t"]) < 2:
            return
        t   = np.array(h["t"])
        rpm = np.array(h["rpm"])
        Te  = np.array(h["Te"])
        I   = np.array(h["I"])

        # Keep only last 10 s for clarity
        mask = t >= max(0, t[-1] - 10.0)
        t, rpm, Te, I = t[mask], rpm[mask], Te[mask], I[mask]

        ax1, ax2, ax3 = self._ax_spd, self._ax_torq, self._ax_cur
        ax1.cla(); ax2.cla(); ax3.cla()

        ax1.plot(t, rpm, color="#1565c0", lw=1.2)
        ax2.plot(t, Te,  color="#c62828", lw=1.2)
        ax3.plot(t, I,   color="#2e7d32", lw=1.2)

        ws_rpm = self._motor.w_sync() * 60.0 / TWO_PI
        ax1.axhline(ws_rpm, color="grey", ls="--", lw=0.8, label="Sync speed")
        ax1.legend(fontsize=7)

        for ax, lbl in zip([ax1, ax2, ax3],
                           ["Speed (RPM)", "Torque (N·m)", "Current (A)"]):
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
        ax3.set_xlabel("Time (s)")

        rpm_now = rpm[-1] if len(rpm) else 0
        Te_now  = Te[-1]  if len(Te)  else 0
        self._sim_status.set(
            f"Status: Running  |  Speed = {rpm_now:.0f} RPM  |  Torque = {Te_now:.2f} N·m"
        )
        self._fig_sim.tight_layout()
        self._canvas_sim.draw()

    # =======================================================================
    # TAB 6 — FAULT CURRENT
    # =======================================================================
    def _tab_fault(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Fault Parameters")
        ctrl.pack(fill="x", padx=8, pady=4)

        for key, lbl, lo, hi in [
            ("Z_src_pu", "Source Impedance Z (pu on S_base)", 0.01, 0.5),
        ]:
            ttk.Label(ctrl, text=lbl, width=38).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=self.v[key],
                      orient="horizontal", length=280).pack(side="left", padx=2)
            ttk.Label(ctrl, textvariable=self.v[key], width=8).pack(side="left")

        ttk.Button(ctrl, text="Run Fault Analysis",
                   style="Accent.TButton",
                   command=self._run_fault).pack(side="right", padx=8)

        self._fig_fault, axes = make_fig(2, 1, figsize=(11, 5))
        self._ax_fault_wave, self._ax_fault_spec = axes
        self._canvas_fault = embed_figure(self._fig_fault, parent, toolbar=True)

        self._fault_info = tk.StringVar(value="Click 'Run Fault Analysis' to compute.")
        ttk.Label(parent, textvariable=self._fault_info,
                  font=("Courier", 9), background="#fff9c4").pack(pady=2)

    def _run_fault(self):
        V    = self.v["V_line"].get()
        f    = self.v["freq"].get()
        Zpu  = self.v["Z_src_pu"].get()
        hp   = self.v["hp"].get()
        S_kva = hp * HP2W / 1000.0

        t, ia, ib, ic, I_sc_rms, I_sc_pk = fault_current(V, f, Zpu, S_kva)

        ax1, ax2 = self._ax_fault_wave, self._ax_fault_spec
        ax1.cla(); ax2.cla()

        ax1.plot(t * 1e3, ia, color="#c62828", lw=1.2, label="Phase A")
        ax1.plot(t * 1e3, ib, color="#2e7d32", lw=1.2, label="Phase B")
        ax1.plot(t * 1e3, ic, color="#1565c0", lw=1.2, label="Phase C")
        ax1.axhline(0, color="k", lw=0.4)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Fault Current (A)")
        ax1.set_title("3-Phase Symmetrical Fault Current (Bolted Fault at Terminals)",
                      fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Current magnitude envelope
        t_env  = np.linspace(0, 0.2, 500)
        tau    = 1.0 / (TWO_PI * f * 0.1)
        dc_env = SQRT2 * I_sc_rms * np.exp(-t_env / tau)
        ax2.plot(t_env * 1e3, SQRT2 * I_sc_rms + dc_env,
                 "r--", lw=1.5, label="Peak envelope")
        ax2.axhline(SQRT2 * I_sc_rms, color="b", ls=":", lw=1.2,
                    label=f"AC peak = {SQRT2*I_sc_rms:.0f} A")
        ax2.axhline(I_sc_rms, color="g", ls="-.", lw=1.2,
                    label=f"RMS = {I_sc_rms:.0f} A")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Current magnitude (A)")
        ax2.set_title("Fault Current Envelope")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        self._fig_fault.tight_layout()
        self._canvas_fault.draw()
        self._fault_info.set(
            f"  Bolted 3φ Fault:  I_sc_rms = {I_sc_rms:.1f} A  |  "
            f"I_sc_peak = {I_sc_pk:.1f} A  |  Z_source = {Zpu:.3f} pu  |  "
            f"Asymmetry factor ≈ {I_sc_pk/(SQRT2*I_sc_rms):.2f}"
        )

    # =======================================================================
    # TAB 7 — PROTECTION COORDINATION
    # =======================================================================
    def _tab_protection(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Relay Settings")
        ctrl.pack(fill="x", padx=8, pady=4)

        settings = [
            ("Ip1",  "Relay 1 — Pickup Current (A)", 5.0,  50.0),
            ("tms1", "Relay 1 — TMS",                0.05,  2.0),
            ("Ip2",  "Relay 2 — Pickup Current (A)", 2.0,  30.0),
            ("tms2", "Relay 2 — TMS",                0.05,  2.0),
        ]
        for key, lbl, lo, hi in settings:
            ttk.Label(ctrl, text=lbl, width=34).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=self.v[key],
                      orient="horizontal", length=180,
                      command=lambda _e: self._plot_protection()).pack(side="left")
            ttk.Label(ctrl, textvariable=self.v[key], width=6).pack(side="left")

        self._fig_prot, ax = make_fig(1, 1, figsize=(10, 5))
        self._ax_prot = ax[0]
        self._canvas_prot = embed_figure(self._fig_prot, parent, toolbar=True)

    def _plot_protection(self):
        Ip1  = self.v["Ip1"].get()
        tms1 = self.v["tms1"].get()
        Ip2  = self.v["Ip2"].get()
        tms2 = self.v["tms2"].get()

        I_arr = np.logspace(np.log10(1.01 * max(Ip1, Ip2)),
                            np.log10(20 * max(Ip1, Ip2)), 300)

        t1_si = tcc_curve(I_arr, Ip1, tms1, 'SI')
        t1_vi = tcc_curve(I_arr, Ip1, tms1, 'VI')
        t1_ei = tcc_curve(I_arr, Ip1, tms1, 'EI')
        t2_si = tcc_curve(I_arr, Ip2, tms2, 'SI')

        ax = self._ax_prot
        ax.cla()
        ax.loglog(I_arr, t1_si, "r-",  lw=2.0, label=f"Main R1 — SI (Ip={Ip1:.1f}A, TMS={tms1:.2f})")
        ax.loglog(I_arr, t1_vi, "r--", lw=1.5, label=f"Main R1 — VI")
        ax.loglog(I_arr, t1_ei, "r:",  lw=1.5, label=f"Main R1 — EI")
        ax.loglog(I_arr, t2_si, "b-",  lw=2.0, label=f"Backup R2 — SI (Ip={Ip2:.1f}A, TMS={tms2:.2f})")

        # Motor starting current (≈ 6 × rated)
        hp   = self.v["hp"].get()
        V    = self.v["V_line"].get()
        pf   = self.v["pf"].get()
        eta  = self.v["eta"].get()
        pol  = self.v["poles"].get()
        r    = motor_params(hp, V, self.v["freq"].get(), pf, eta, pol)
        I_rated  = r["I_line"]
        I_start  = 6.0 * I_rated
        ax.axvline(I_start, color="orange", ls="--", lw=1.5,
                   label=f"Motor start current ≈ {I_start:.1f} A")
        ax.axvline(I_rated, color="green",  ls="-.", lw=1.2,
                   label=f"Rated current = {I_rated:.1f} A")

        ax.set_xlabel("Current (A)", fontsize=9)
        ax.set_ylabel("Time (s)",    fontsize=9)
        ax.set_title("Protection Coordination — Time-Current Characteristics",
                     fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(0.01, 100)
        self._fig_prot.tight_layout()
        self._canvas_prot.draw()

    # =======================================================================
    # TAB 8 — SPEED CONTROLLER
    # =======================================================================
    def _tab_speed_ctrl(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Controller Settings")
        ctrl.pack(fill="x", padx=8, pady=4)

        sliders = [
            ("speed_ref", "Speed Reference (RPM)", 200.0, 1800.0),
            ("kp",        "PID — Kp",               0.1,    10.0),
            ("ki",        "PID — Ki",                0.0,     5.0),
            ("kd",        "PID — Kd",                0.0,     2.0),
            ("step_load", "Step Load Torque (N·m)",  0.0,    50.0),
        ]
        for key, lbl, lo, hi in sliders:
            ttk.Label(ctrl, text=lbl, width=28).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=self.v[key],
                      orient="horizontal", length=160).pack(side="left", padx=2)
            ttk.Label(ctrl, textvariable=self.v[key], width=7).pack(side="left")

        mode_frame = ttk.Frame(ctrl)
        mode_frame.pack(side="right", padx=6)
        self._ctrl_mode = tk.StringVar(value="PID")
        ttk.Radiobutton(mode_frame, text="PID",   variable=self._ctrl_mode,
                        value="PID").pack(side="left")
        ttk.Radiobutton(mode_frame, text="Fuzzy", variable=self._ctrl_mode,
                        value="Fuzzy").pack(side="left")

        btn = ttk.Frame(ctrl)
        btn.pack(side="right", padx=8)
        ttk.Button(btn, text="▶ Run", style="Accent.TButton",
                   command=self._ctrl_run).pack(side="left", padx=3)
        ttk.Button(btn, text="■ Stop",
                   command=self._ctrl_stop).pack(side="left", padx=3)
        ttk.Button(btn, text="↺ Reset",
                   command=self._ctrl_reset).pack(side="left", padx=3)

        self._fig_ctrl, axes = make_fig(3, 1, figsize=(11, 7))
        self._ax_cspd, self._ax_ctrq, self._ax_calpha = axes
        self._canvas_ctrl = embed_figure(self._fig_ctrl, parent, toolbar=False)

        self._ctrl_status = tk.StringVar(value="Status: Idle")
        ttk.Label(parent, textvariable=self._ctrl_status).pack(pady=2)

        self._ctrl_motor = MotorModel()
        self._pid = PIDController()
        self._fuzzy = FuzzyController()

    def _ctrl_run(self):
        if self._ctrl_running:
            return
        self._ctrl_running = True
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)
        self._ctrl_thread.start()

    def _ctrl_stop(self):
        self._ctrl_running = False

    def _ctrl_reset(self):
        self._ctrl_running = False
        time.sleep(0.05)
        self._ctrl_motor.reset()
        self._pid.reset()
        self._fuzzy._prev_err = 0.0
        for ax in [self._ax_cspd, self._ax_ctrq, self._ax_calpha]:
            ax.cla()
            ax.grid(True, alpha=0.3)
        self._canvas_ctrl.draw()
        self._ctrl_status.set("Status: Reset")

    def _ctrl_loop(self):
        self._ctrl_motor.reset()
        self._pid.kp = self.v["kp"].get()
        self._pid.ki = self.v["ki"].get()
        self._pid.kd = self.v["kd"].get()
        self._pid.reset()
        self._fuzzy._prev_err = 0.0

        V     = self.v["V_line"].get()
        alpha = 90.0           # start mid-range
        dt    = 0.004

        # Scale factor: maps PID output (RPM units) to firing-angle rate (°/s).
        # Lower α → higher V_rms → higher speed, so positive error reduces α.
        ALPHA_SCALE = 0.01  # °·s / RPM

        while self._ctrl_running:
            sp    = self.v["speed_ref"].get()
            meas  = self._ctrl_motor.hist["rpm"][-1] if self._ctrl_motor.hist["rpm"] else 0.0

            # Step load at t = 3 s
            t_now = self._ctrl_motor.t
            Tl = self.v["step_load"].get() if t_now >= 3.0 else self.v["T_load"].get()

            mode = self._ctrl_mode.get()
            if mode == "PID":
                self._pid.kp = self.v["kp"].get()
                self._pid.ki = self.v["ki"].get()
                self._pid.kd = self.v["kd"].get()
                # PID output u > 0 when speed < setpoint.
                # Correct action: decrease α (increase voltage → increase speed).
                u  = self._pid(sp, meas)
                da = u * ALPHA_SCALE          # α rate [°/s]
                alpha = float(np.clip(alpha - da * dt, 20.0, 175.0))
            else:
                da    = self._fuzzy(sp, meas, dt=dt)
                alpha = float(np.clip(alpha - da * dt, 20.0, 175.0))

            self._ctrl_motor.step(V, alpha, Tl, dt=dt)
            if len(self._ctrl_motor.hist["t"]) % 15 == 0:
                self.after(0, self._ctrl_update_plot)
            time.sleep(dt)

        self.after(0, lambda: self._ctrl_status.set("Status: Stopped"))

    def _ctrl_update_plot(self):
        h = self._ctrl_motor.hist
        if len(h["t"]) < 2:
            return
        t     = np.array(h["t"])
        rpm   = np.array(h["rpm"])
        Te    = np.array(h["Te"])
        alpha = np.array(h["alpha"])

        mask = t >= max(0, t[-1] - 12.0)
        t, rpm, Te, alpha = t[mask], rpm[mask], Te[mask], alpha[mask]

        ax1, ax2, ax3 = self._ax_cspd, self._ax_ctrq, self._ax_calpha
        ax1.cla(); ax2.cla(); ax3.cla()

        sp = self.v["speed_ref"].get()
        ax1.plot(t, rpm,   color="#1565c0", lw=1.2, label="Speed")
        ax1.axhline(sp,  color="r",    ls="--", lw=1.0, label=f"Ref = {sp:.0f} RPM")
        ax2.plot(t, Te,    color="#c62828", lw=1.2, label="Torque")
        ax3.plot(t, alpha, color="#6a1b9a", lw=1.2, label="Firing angle α")

        mode = self._ctrl_mode.get()
        ax1.set_title(f"Speed Controller ({mode}) — Step Load at t = 3 s",
                      fontweight="bold")
        for ax, lbl in zip([ax1, ax2, ax3],
                           ["Speed (RPM)", "Torque (N·m)", "Firing Angle (°)"]):
            ax.set_ylabel(lbl)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        ax3.set_xlabel("Time (s)")

        rpm_now = rpm[-1] if len(rpm) else 0
        self._ctrl_status.set(
            f"Status: Running [{mode}]  |  Speed = {rpm_now:.0f} RPM  |  "
            f"Ref = {sp:.0f} RPM  |  Error = {sp - rpm_now:.0f} RPM"
        )
        self._fig_ctrl.tight_layout()
        self._canvas_ctrl.draw()

    # =======================================================================
    # TAB 9 — THERMAL & ECONOMIC ANALYSIS
    # =======================================================================
    def _tab_thermal_econ(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Analysis Parameters")
        ctrl.pack(fill="x", padx=8, pady=4)

        for key, lbl, lo, hi in [
            ("R_th",       "Thermal Resistance R_th (°C/W)", 0.1, 5.0),
            ("hours",      "Operating Hours / Year",        1000, 8760),
            ("energy_cost","Energy Cost ($/kWh)",            0.04, 0.30),
        ]:
            ttk.Label(ctrl, text=lbl, width=34).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=self.v[key],
                      orient="horizontal", length=200,
                      command=lambda _e: self._plot_thermal()).pack(side="left")
            ttk.Label(ctrl, textvariable=self.v[key], width=7).pack(side="left")

        ttk.Button(ctrl, text="Calculate", style="Accent.TButton",
                   command=self._plot_thermal).pack(side="right", padx=8)

        self._fig_therm, axes = make_fig(1, 2, figsize=(12, 5))
        self._ax_temp, self._ax_econ = axes
        self._canvas_therm = embed_figure(self._fig_therm, parent, toolbar=True)

    def _plot_thermal(self):
        hp   = self.v["hp"].get()
        V    = self.v["V_line"].get()
        f    = self.v["freq"].get()
        pf   = self.v["pf"].get()
        eta  = self.v["eta"].get()
        pol  = self.v["poles"].get()
        r    = motor_params(hp, V, f, pf, eta, pol)

        R_th   = self.v["R_th"].get()
        hours  = self.v["hours"].get()
        e_cost = self.v["energy_cost"].get()

        P_loss = r["P_in"] - r["P_out"]   # core + copper losses
        T_amb  = 25.0
        self._thermal.R_th = R_th

        # Temperature rise curve (time domain)
        dt   = 10.0   # 10 s steps
        t_th = np.arange(0, 3600 * 2, dt)   # 2 hours
        C_th = 500.0
        tau  = R_th * C_th
        T_ss = T_amb + P_loss * R_th
        T_t  = T_ss - (T_ss - T_amb) * np.exp(-t_th / (tau + 1e-9))

        ax1 = self._ax_temp
        ax1.cla()
        ax1.plot(t_th / 60, T_t, color="#c62828", lw=2)
        ax1.axhline(T_ss, color="grey", ls="--", lw=1.2,
                    label=f"T_steady = {T_ss:.1f} °C")
        ax1.axhline(130, color="orange", ls=":", lw=1.5, label="Class B limit 130°C")
        ax1.axhline(155, color="red",    ls=":", lw=1.5, label="Class F limit 155°C")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title("Motor Temperature Rise", fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(20, max(200, T_ss + 30))

        # Economic analysis — annual energy cost vs load
        ax2 = self._ax_econ
        ax2.cla()
        loads = np.linspace(0.1, 1.5, 50)
        kva_vals, cost_vals = [], []
        for frac in loads:
            P_o  = frac * r["P_out"]
            P_i  = P_o / max(eta, 0.01)
            kva_vals.append(P_i / 1000)
            annual_kwh = P_i / 1000 * hours
            cost_vals.append(annual_kwh * e_cost)

        ax2.plot(loads * 100, cost_vals, color="#1565c0", lw=2)
        ax2.axvline(100, color="r", ls="--", lw=1.2, label="Rated load")
        at_full  = r["P_in"] / 1000 * hours * e_cost
        ax2.scatter([100], [at_full], color="red", s=60, zorder=5,
                    label=f"Full-load cost = ${at_full:.0f}/yr")
        ax2.set_xlabel("Load Level (%)")
        ax2.set_ylabel("Annual Energy Cost ($)")
        ax2.set_title("Annual Energy Cost vs Load Level", fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        self._fig_therm.tight_layout()
        self._canvas_therm.draw()

    # =======================================================================
    # TAB 10 — HARMONICS & POWER QUALITY
    # =======================================================================
    def _tab_harmonics(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Harmonic Analysis Parameters")
        ctrl.pack(fill="x", padx=8, pady=4)

        for key, lbl, lo, hi in [
            ("alpha",   "Firing Angle α (°)", 0.0, 175.0),
            ("pf_load", "Load PF",             0.5,   1.0),
        ]:
            ttk.Label(ctrl, text=lbl, width=22).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=self.v[key],
                      orient="horizontal", length=220,
                      command=lambda _e: self._plot_harmonics()).pack(side="left")
            ttk.Label(ctrl, textvariable=self.v[key], width=7).pack(side="left")

        ttk.Button(ctrl, text="Analyse", style="Accent.TButton",
                   command=self._plot_harmonics).pack(side="right", padx=8)

        self._fig_harm, axes = make_fig(2, 2, figsize=(12, 6))
        self._ax_hv, self._ax_hi, self._ax_hs_v, self._ax_hs_i = axes
        self._canvas_harm = embed_figure(self._fig_harm, parent, toolbar=True)

        self._harm_status = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._harm_status,
                  font=("Courier", 9), background="#e8f5e9").pack(pady=2)

    def _plot_harmonics(self):
        alpha   = self.v["alpha"].get()
        pf_load = self.v["pf_load"].get()
        V       = self.v["V_line"].get()
        f       = self.v["freq"].get()

        t, v_s, v_out, i_out = controller_waveform(V, f, alpha, pf_load, cycles=10)
        dt = t[1] - t[0]

        freqs_v, mag_v, thd_v = fft_analysis(v_out, dt)
        freqs_i, mag_i, thd_i = fft_analysis(i_out, dt)

        ax1, ax2, ax3, ax4 = (self._ax_hv, self._ax_hi,
                               self._ax_hs_v, self._ax_hs_i)
        ax1.cla(); ax2.cla(); ax3.cla(); ax4.cla()

        # Time-domain waveforms
        ms_limit = min(3.0 / f * 1e3, t[-1] * 1e3)
        mask_t = (t * 1e3) <= ms_limit
        ax1.plot(t[mask_t]*1e3, v_out[mask_t], color="#c62828", lw=1.5)
        ax1.set_title(f"Output Voltage  (α={alpha:.0f}°, PF={pf_load:.2f})",
                      fontweight="bold")
        ax1.set_ylabel("Voltage (V)"); ax1.set_xlabel("Time (ms)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(t[mask_t]*1e3, i_out[mask_t], color="#2e7d32", lw=1.5)
        ax2.set_title("Output Current  (normalised)")
        ax2.set_ylabel("Current (pu)"); ax2.set_xlabel("Time (ms)")
        ax2.grid(True, alpha=0.3)

        # Harmonic spectra (up to 20th harmonic)
        h_freqs  = np.arange(1, 21) * f
        h_mag_v  = []
        h_mag_i  = []
        tol = 3.0
        for hf in h_freqs:
            mv = mag_v[np.abs(freqs_v - hf) < tol]
            mi = mag_i[np.abs(freqs_i - hf) < tol]
            h_mag_v.append(mv.max() if len(mv) else 0.0)
            h_mag_i.append(mi.max() if len(mi) else 0.0)

        h_orders = np.arange(1, 21)
        clrs_v = ["#c62828" if h == 1 else "#ef9a9a" for h in h_orders]
        clrs_i = ["#1565c0" if h == 1 else "#90caf9" for h in h_orders]

        ax3.bar(h_orders, h_mag_v, color=clrs_v, edgecolor="k", lw=0.5)
        ax3.set_xlabel("Harmonic Order"); ax3.set_ylabel("Amplitude (V)")
        ax3.set_title(f"Voltage Harmonic Spectrum  (THD = {thd_v:.1f} %)",
                      fontweight="bold")
        ax3.set_xticks(h_orders)
        ax3.grid(True, axis="y", alpha=0.3)

        ax4.bar(h_orders, h_mag_i, color=clrs_i, edgecolor="k", lw=0.5)
        ax4.set_xlabel("Harmonic Order"); ax4.set_ylabel("Amplitude (pu)")
        ax4.set_title(f"Current Harmonic Spectrum  (THD = {thd_i:.1f} %)",
                      fontweight="bold")
        ax4.set_xticks(h_orders)
        ax4.grid(True, axis="y", alpha=0.3)

        self._fig_harm.tight_layout()
        self._canvas_harm.draw()
        self._harm_status.set(
            f"  Voltage THD = {thd_v:.2f} %    |    "
            f"Current THD = {thd_i:.2f} %    |    "
            f"α = {alpha:.1f}°    |    PF_load = {pf_load:.3f}"
        )

    # =======================================================================
    # GLOBAL UPDATE
    # =======================================================================
    def _update_all(self):
        self._update_param_labels()
        self._refresh_calc()
        self._plot_waveform()
        self._plot_protection()
        self._plot_thermal()
        self._plot_harmonics()

    def _on_resize(self, event):
        # Only respond to root window resize
        if event.widget is self:
            for attr in ["_canvas_overview", "_canvas_wave", "_canvas_sim",
                         "_canvas_fault", "_canvas_prot", "_canvas_ctrl",
                         "_canvas_therm", "_canvas_harm"]:
                try:
                    canvas = getattr(self, attr)
                    canvas.figure.tight_layout()
                    canvas.draw()
                except Exception:
                    pass

    def _on_close(self):
        self._sim_running  = False
        self._ctrl_running = False
        time.sleep(0.1)
        self.destroy()


# ===========================================================================
# FORMATTED RESULTS TEXT
# ===========================================================================

PROBLEM_TEXT = """\
PROBLEM: 3φ Delta-Connected AC Voltage Controller — Speed Control of Induction Motor
────────────────────────────────────────────────────────────────────────────────────
Motor: 3-phase, 5 hp, 208 V (line-to-line), 60 Hz, delta (Δ) connected.
Controller: 3-phase AC voltage controller (anti-parallel thyristor pairs in each line).
Full-load operating point: Output = 5 hp, V_line = 208 V, PF = 0.85 lagging, η = 90 %.

(a) CIRCUIT:  Three anti-parallel thyristor pairs (one pair per supply line) connect
    the 3-phase AC source to the delta-connected induction motor.  Gate pulses for
    each pair are displaced by 120° to maintain balanced 3-phase operation.
    The firing angle α controls the RMS voltage applied to the motor, thereby
    controlling its speed by adjusting the air-gap flux and the torque.

(b) INPUT kVA:  P_out = 5 × 746 = 3730 W
                P_in  = P_out / η = 3730 / 0.90 = 4144 W
                S_in  = P_in  / PF = 4144 / 0.85 ≈ 4876 VA  ≈ 4.876 kVA

(c) FIRING ANGLE RANGE:  For delta-connected resistive-inductive load the minimum
    usable firing angle equals the load power-factor angle φ = cos⁻¹(0.85) ≈ 31.8°.
    Below this angle the output cannot be further increased (thyristor always on).
    For full-load the controller must deliver rated voltage, so α_min ≈ φ ≈ 31.8°.
    Reducing speed requires increasing α up to the practical maximum of ≈ 150°–160°.
    Full-load firing angle range: 31.8° ≤ α ≤ 180°.

(d) THYRISTOR RATINGS:
    I_line    = S_in / (√3 × V_L) = 4876 / (1.732 × 208) ≈ 13.5 A
    I_phase Δ = I_line / √3 ≈ 7.8 A  (current through each thyristor pair)
    V_thyristor (PIV) = √2 × V_L = 1.414 × 208 ≈ 294 V
    I_thy_rms  = I_phase / √2 ≈ 5.5 A   (each thyristor conducts half cycle)
    I_thy_peak = √2 × I_phase ≈ 11.0 A
    Use device ratings:  V_T ≥ 294 V (add 2× safety → ≥ 600 V),
                         I_T ≥ 5.5 A rms / 11 A peak.

(e) WAVEFORM (PF = 0.80, α = 60°):  See the Waveforms tab.
    φ = cos⁻¹(0.80) = 36.87° < α = 60°, so thyristors fire AFTER natural
    zero-crossing of their respective commutation voltages.
    Each thyristor conducts from ωt = α to ωt = β (extinction angle ≈ π+φ).
    The motor voltage follows the source during conduction and is zero during
    blocking.  The current lags the applied voltage by φ and decays
    exponentially during the blocking intervals (inductive load effect).
"""


def format_results(hp, V, f, pf, eta, poles, r):
    sep  = "─" * 70 + "\n"
    out  = []
    out.append("3φ DELTA-CONNECTED AC VOLTAGE CONTROLLER — FULL CALCULATION RESULTS\n")
    out.append(sep)
    out.append(f"  INPUT PARAMETERS\n")
    out.append(f"    Rated power     : {hp:.1f} hp  = {hp*HP2W:.0f} W\n")
    out.append(f"    Line voltage    : {V:.0f} V  (delta connection → V_phase = V_line)\n")
    out.append(f"    Frequency       : {f:.0f} Hz\n")
    out.append(f"    Power factor    : {pf:.3f} lagging\n")
    out.append(f"    Efficiency      : {eta*100:.1f} %\n")
    out.append(f"    Poles           : {poles}\n")
    out.append(sep)
    out.append(f"  (b) APPARENT INPUT POWER (kVA)\n")
    out.append(f"    P_out = {r['P_out']:.0f} W\n")
    out.append(f"    P_in  = P_out / η = {r['P_out']:.0f} / {eta:.2f} = {r['P_in']:.0f} W\n")
    out.append(f"    Q_in  = {r['Q_in']:.0f} VAR  (reactive power)\n")
    out.append(f"  ► S_in  = P_in / PF = {r['P_in']:.0f} / {pf:.2f} = {r['S_in']:.0f} VA"
               f"  ≈  {r['S_in']/1000:.3f} kVA\n")
    out.append(sep)
    out.append(f"  (c) FIRING ANGLE RANGE\n")
    out.append(f"    φ = cos⁻¹({pf:.2f}) = {r['phi_deg']:.2f}°  (load power-factor angle)\n")
    out.append(f"    For RL load the thyristor cannot fire before natural commutation\n")
    out.append(f"    voltage zero-crossing, so  α_min = φ = {r['alpha_min']:.2f}°\n")
    out.append(f"  ► Firing angle range: {r['alpha_min']:.2f}° ≤ α ≤ {r['alpha_max']:.1f}°\n")
    out.append(sep)
    out.append(f"  (d) THYRISTOR RATINGS\n")
    out.append(f"    3-phase line current  I_L = S / (√3·V_L) = {r['I_line']:.3f} A\n")
    out.append(f"    Delta phase current   I_ph = I_L / √3    = {r['I_phase']:.3f} A\n")
    out.append(f"    Peak phase current    I_pk = √2 · I_ph   = {r['I_thy_pk']:.3f} A\n")
    out.append(f"  ► Thyristor voltage (PIV) = √2 · V_L       = {r['V_thy']:.2f} V\n")
    out.append(f"  ► Thyristor current (rms) = I_ph / √2      = {r['I_thy_rm']:.3f} A\n")
    out.append(f"  ► Thyristor current (avg) = I_pk / π       = {r['I_thy_av']:.3f} A\n")
    out.append(f"    Recommended device: V_T ≥ 600 V, I_T ≥ {r['I_thy_rm']*1.5:.1f} A rms\n")
    out.append(sep)
    out.append(f"  ADDITIONAL QUANTITIES\n")
    out.append(f"    Synchronous speed : {r['n_sync']:.0f} RPM  ({r['w_sync']:.2f} rad/s)\n")
    out.append(f"    Full-load torque  : {r['T_full']:.2f} N·m\n")
    out.append(f"    Input power factor angle φ = {r['phi_deg']:.2f}°\n")
    out.append(sep)
    out.append(f"  (e) WAVEFORM NOTE (PF=0.80, α=60°)\n")
    out.append(f"    φ_load = cos⁻¹(0.80) = 36.87°  < α = 60°\n")
    out.append(f"    Each thyristor fires at 60° and extinguishes near ωt ≈ π+φ ≈ 216.9°.\n")
    out.append(f"    The motor voltage equals the source voltage during conduction\n")
    out.append(f"    and zero during blocking intervals.\n")
    out.append(f"    The current waveform shows inductive lag and exponential decay\n")
    out.append(f"    during OFF intervals. See the Waveforms tab for the plot.\n")
    return "".join(out)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()
