# src/multi_ring.py
from __future__ import annotations
import numpy as np
import magpylib as magpy
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings

# ----------------------------
# Config dataclasses
# ----------------------------
@dataclass
class MagnetParams:
    radius: float = 0.02        # m
    height: float = 0.02        # m
    magnetization: float = 1000 # A/m
    mass: float = 0.05          # kg

@dataclass
class TubeParams:
    ring_radius: float = 0.05             # m (tube inner radius)
    ring_count: int = 20                  # number of discrete rings along tube length
    ring_length: float = 0.5              # total axial length of tube (m)
    radial_integration_points: int = 200  # points per ring for flux calc
    wire_radius: float = 1e-3             # conductor wire radius for L estimate (m)

@dataclass
class ElectricParams:
    resistance_per_ring: float = 1.44e-3  # ohms (can be scalar or array length N)
    include_mutual_inductance: bool = False  # whether to compute L_ij mutuals (not auto)
    # if include_mutual_inductance==False we will use diagonal L only

@dataclass
class SimConfig:
    dt: float = 1e-4
    total_time: float = 0.5
    initial_height: float = 0.2
    initial_velocity: float = 0.0
    gravity: float = 9.81
    gradient_step: float = 1e-4  # for dB/dz used in force estimate
    atol: float = 1e-8
    rtol: float = 1e-6

# ----------------------------
# Multi-ring simulation class
# ----------------------------
class MultiRingSimulation:
    def __init__(
        self,
        magnet: MagnetParams | None = None,
        tube: TubeParams | None = None,
        electric: ElectricParams | None = None,
        config: SimConfig | None = None,
    ):
        self.mag = magnet or MagnetParams()
        self.tube = tube or TubeParams()
        self.elec = electric or ElectricParams()
        self.cfg = config or SimConfig()

        # create magnet (cylinder) in magpylib
        self._magnet = magpy.magnet.Cylinder(
            magnetization=(0, 0, self.mag.magnetization),
            dimension=(2.0 * self.mag.radius, self.mag.height),
            position=(0.0, 0.0, self.cfg.initial_height),
        )

        # build rings positions along tube axis (centered around z=0)
        N = int(self.tube.ring_count)
        Ltot = self.tube.ring_length
        # distribute rings uniformly from z = -Ltot/2 to +Ltot/2
        self.ring_z = np.linspace(-Ltot/2, Ltot/2, N)
        self.N = N

        # create magpy current objects for rings (for field eval)
        self.rings = []
        for zi in self.ring_z:
            if hasattr(magpy.current, "Loop"):
                ring = magpy.current.Loop(diameter=2.0*self.tube.ring_radius, current=0.0, position=(0,0,zi))
            else:
                ring = magpy.current.Circle(diameter=2.0*self.tube.ring_radius, current=0.0, position=(0,0,zi))
            self.rings.append(ring)

        # compute dipole moment of cylinder (for force estimate)
        volume = np.pi * self.mag.radius**2 * self.mag.height
        self._dipole_moment = self.mag.magnetization * volume

        # Resistances: allow vector or scalar
        if np.isscalar(self.elec.resistance_per_ring):
            self.R = np.eye(N) * float(self.elec.resistance_per_ring)
        else:
            arr = np.asarray(self.elec.resistance_per_ring, dtype=float)
            assert arr.shape[0] == N
            self.R = np.diag(arr)

        # Inductance matrix L: by default approximate diagonal only
        self.L = self._approx_inductance_matrix()

        # precompute radial samples for flux integration (same radii for all rings)
        nrad = int(self.tube.radial_integration_points)
        self.radii = np.linspace(0.0, self.tube.ring_radius, nrad+1)
        self.sample_radii = 0.5*(self.radii[:-1] + self.radii[1:])
        self.radial_widths = self.radii[1:] - self.radii[:-1]

    def _approx_inductance_matrix(self) -> np.ndarray:
        """Approximate L as diagonal using the single-loop formula.
           Optionally extend to mutuals if user provides them externally.
        """
        a = self.tube.ring_radius
        rho = self.tube.wire_radius
        mu0 = 4e-7 * np.pi
        if rho <= 0:
            warnings.warn("wire_radius <= 0, setting L=0")
            Ldiag = np.zeros(self.N)
        else:
            Ldiag = mu0 * a * (np.log(8.0 * a / rho) - 2.0) * np.ones(self.N)
        Lmat = np.diag(Ldiag)
        # TODO: if include_mutual_inductance: compute or accept an L_mat from user
        return Lmat

    # ---- flux computation for a vector of ring positions ----
    def _compute_flux_vector(self, magnet_z: float) -> np.ndarray:
        """Compute flux through each ring due to magnet at magnet_z.
           Uses axisymmetric radial integration per ring (discrete rings at different z).
        """
        self._magnet.position = (0.0, 0.0, magnet_z)
        # sensors for each ring: create sample points on that ring's disk plane
        N = self.N
        bzs = np.zeros((N, self.sample_radii.shape[0]))  # store Bz samples per ring

        # We'll build a single array of observation points to query magpy once (more efficient)
        obs = []
        ring_indices = []
        for i, zi in enumerate(self.ring_z):
            # sensors at (r * cos, r * sin, zi)
            rs = self.sample_radii
            thetas = np.zeros_like(rs)  # all at angle 0 for Bz (symmetry ensures same Bz at any theta)
            xs = rs
            ys = np.zeros_like(rs)
            zs = np.full_like(rs, zi)
            # append points
            for x,y,z in zip(xs, ys, zs):
                obs.append((x,y,z))
                ring_indices.append(i)
        obs = np.array(obs, dtype=float)
        # get B field at all observation points
        b_all = np.asarray(magpy.getB(self._magnet, obs), dtype=float)
        # accumulate Bz into bzs by ring
        idx = 0
        samples_per_ring = self.sample_radii.shape[0]
        for i in range(N):
            slice_bz = b_all[idx:idx+samples_per_ring, 2]  # Bz
            bzs[i,:] = slice_bz
            idx += samples_per_ring

        # integrate radially: Φ_i = sum_k Bz(r_k)*2π r_k Δr_k
        area_elems = 2.0 * np.pi * self.sample_radii * self.radial_widths
        fluxes = np.sum(bzs * area_elems[np.newaxis,:], axis=1)
        return fluxes  # shape (N,)

    def _compute_flux_derivative_vec(self, z: float) -> np.ndarray:
        """Compute Φ'(z) ≈ dΦ/dz via central finite difference using gradient_step."""
        dz = self.cfg.gradient_step
        Phi_plus = self._compute_flux_vector(z + dz)
        Phi_minus = self._compute_flux_vector(z - dz)
        return (Phi_plus - Phi_minus) / (2.0 * dz)

    # ---- ODE system for solve_ivp ----
    def _ode_rhs(self, t, y):
        """
        y = [I_0..I_{N-1}, z, v]
        returns dy/dt
        """
        N = self.N
        I = y[0:N]
        z = float(y[N])
        v = float(y[N+1])

        # compute dPhi/dz vector at current z
        Phi_prime = self._compute_flux_derivative_vec(z)  # shape (N,)

        # RL matrix form: L dI/dt + R I = -Phi'(z) * v
        # => dI/dt = L^{-1} ( -R I - Phi'(z) v )
        # solve L x = RHS (avoid explicit inverse)
        RHS = - (self.R.dot(I) + Phi_prime * v)
        try:
            dI_dt = np.linalg.solve(self.L, RHS)
        except np.linalg.LinAlgError:
            # fallback: if L singular, use pseudo-inverse (numerically stable-ish)
            Lpinv = np.linalg.pinv(self.L)
            dI_dt = Lpinv.dot(RHS)

        # magnetic force: F = I^T Phi'(z)
        Fmag = float(I.dot(Phi_prime))

        # acceleration (sign conv: +z up, gravity down)
        acc = -self.cfg.gravity + (Fmag / self.mag.mass)

        dz_dt = v
        dv_dt = acc

        dy = np.zeros_like(y)
        dy[0:N] = dI_dt
        dy[N] = dz_dt
        dy[N+1] = dv_dt
        return dy

    def run(self):
        """Integrate system and return records similar to previous code."""
        N = self.N
        t0 = 0.0
        tf = self.cfg.total_time
        # initial state: zero currents, initial z and v
        y0 = np.zeros(N + 2)
        y0[N] = self.cfg.initial_height
        y0[N+1] = self.cfg.initial_velocity

        sol = solve_ivp(
            fun=self._ode_rhs,
            t_span=(t0, tf),
            y0=y0,
            method='RK45',
            atol=self.cfg.atol,
            rtol=self.cfg.rtol,
            dense_output=False,
            max_step=self.cfg.dt
        )

        # build records
        records = []
        for k, t in enumerate(sol.t):
            yk = sol.y[:, k]
            I = yk[0:N]
            z = float(yk[N])
            v = float(yk[N+1])
            Phi = self._compute_flux_vector(z)
            Phi_prime = self._compute_flux_derivative_vec(z)
            emf = -Phi_prime * v
            Fmag = float(I.dot(Phi_prime))
            acc = -self.cfg.gravity + (Fmag / self.mag.mass)

            records.append({
                "time": t,
                "position": z,
                "velocity": v,
                "acceleration": acc,
                "flux": Phi,
                "emf": emf,
                "current": I.copy(),
                "magnetic_force": Fmag,
            })

        return records

# ----------------------------
# Helper plotting (to mimic previous outputs)
# ----------------------------
def plot_records(records):
    import pandas as pd
    # Convert records into DataFrame columns similar to previous format:
    t = np.array([r["time"] for r in records])
    z = np.array([r["position"] for r in records])
    v = np.array([r["velocity"] for r in records])
    a = np.array([r["acceleration"] for r in records])
    F = np.array([r["magnetic_force"] for r in records])
    # For flux/emf/current we have vectors per record; pick flux of central ring (or sum)
    flux_arr = np.array([np.sum(r["flux"]) for r in records])  # total flux sum over rings
    emf_arr = np.array([np.sum(r["emf"]) for r in records])
    # For current choose max absolute current across rings as representative
    current_max = np.array([np.max(np.abs(r["current"])) for r in records])

    fig, axes = plt.subplots(nrows=7, ncols=1, sharex=True, figsize=(10, 14))
    axes[0].plot(t, z); axes[0].set_ylabel("Position (m)"); axes[0].set_title("Magnet Position", loc="left")
    axes[1].plot(t, v); axes[1].set_ylabel("Velocity (m/s)"); axes[1].set_title("Magnet Velocity", loc="left")
    axes[2].plot(t, a); axes[2].set_ylabel("Acceleration (m/s²)"); axes[2].set_title("Magnet Acceleration", loc="left")
    axes[3].plot(t, F); axes[3].set_ylabel("Force (N)"); axes[3].set_title("Magnetic Force", loc="left")
    axes[4].plot(t, emf_arr); axes[4].set_ylabel("EMF (V)"); axes[4].set_title("Induced EMF (sum over rings)", loc="left")
    axes[5].plot(t, current_max); axes[5].set_ylabel("Current (A)"); axes[5].set_title("Max Induced Current", loc="left")
    axes[6].plot(t, flux_arr); axes[6].set_ylabel("Flux (T·m²)"); axes[6].set_title("Total Flux (sum rings)", loc="left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Falling Magnet in Multi-Ring Tube: Kinematics & Electromagnetics for 7 rings", y=0.995)
    fig.tight_layout()
    plt.savefig("graficos/7anillo.png")
    plt.show()