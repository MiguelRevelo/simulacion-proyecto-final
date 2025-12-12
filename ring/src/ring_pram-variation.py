from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import magpylib as magpy
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Configuration Data Classes
# ----------------------------
@dataclass
class MagnetParams:
    """Material and geometric properties of the falling magnet."""
    radius: float = 0.02        # meters
    height: float = 0.02        # meters
    magnetization: float = 1000  # A/m along +z
    mass: float = 0.05          # kg


@dataclass
class RingParams:
    """Electrical and geometric properties of the stationary ring."""
    radius: float = 0.05            # meters
    resistance: float = 1.44e-3         # ohms
    wire_radius: float = 1e-3       # meters (assumed round wire)
    radial_integration_points: int = 1000


@dataclass
class SimulationConfig:
    """Integration configuration and kinematic initial conditions."""
    time_step: float = 1e-3       # seconds
    total_time: float = 0.4       # seconds
    initial_height: float = 0.2   # meters (above the ring plane)
    initial_velocity: float = 0.0 # m/s (positive = upward)
    gravity: float = 9.81         # m/s^2
    gradient_step: float = 1e-4   # meters for finite-difference gradient


# ----------------------------
# Simulation
# ----------------------------

class FallingMagnetSimulation:
    """Numerically integrates a magnet falling through a conductive ring."""

    def __init__(
        self,
        magnet_params: MagnetParams | None = None,
        ring_params: RingParams | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.magnet_params = magnet_params or MagnetParams()
        self.ring_params = ring_params or RingParams()
        self.config = config or SimulationConfig()

        # Magnet source
        self._magnet = magpy.magnet.Cylinder(
            magnetization=(0.0, 0.0, self.magnet_params.magnetization),
            dimension=(2.0 * self.magnet_params.radius, self.magnet_params.height),
            position=(0.0, 0.0, self.config.initial_height),
        )

        # Ring source 
        if hasattr(magpy.current, "Loop"):
            self._ring = magpy.current.Loop(
                diameter=2.0 * self.ring_params.radius,
                current=0.0,
                position=(0.0, 0.0, 0.0),
            )
        else:
            self._ring = magpy.current.Circle(
                diameter=2.0 * self.ring_params.radius,
                current=0.0,
                position=(0.0, 0.0, 0.0),
            )

        # Magnetic dipole moment of the cylinder (m = M * V)
        volume = np.pi * self.magnet_params.radius**2 * self.magnet_params.height
        self._dipole_moment = self.magnet_params.magnetization * volume  # [A·m^2]

        # Single-loop self inductance (approx.)
        mu0 = 4e-7 * np.pi
        a = self.ring_params.radius
        rho = self.ring_params.wire_radius
        # L ≈ μ0 a [ln(8a/ρ) - 2]
        self._L = mu0 * a * (np.log(8.0 * a / rho) - 2.0) if rho > 0 else 0.0

    def run(self) -> List[Dict[str, float | Tuple[float, float, float]]]:
        """Simulate the magnet dynamics and return time-series data."""
        dt = self.config.time_step
        steps = int(np.floor(self.config.total_time / dt))

        time = 0.0
        position = self.config.initial_height
        velocity = self.config.initial_velocity

        # Precompute initial flux and init current
        flux_last = self._compute_flux(position)
        current = 0.0  # induced current in the ring

        records: List[Dict[str, float | Tuple[float, float, float]]] = []

        for step in range(steps):
            # Flux and EMF
            flux_current = self._compute_flux(position)
            emf = -(flux_current - flux_last) / dt if step > 0 else 0.0  # -dΦ/dt

            # RL update for current (semi-analytic exact for linear term)
            L = self._L
            R = self.ring_params.resistance
            if L > 0 and R > 0:
                alpha = np.exp(-R * dt / L)
                # assumes emf roughly constant over the step
                current = current * alpha + (1.0 - alpha) * (emf / R)
            elif R > 0:
                current = emf / R
            else:
                # Avoid division by zero; if R==0 this is a superconducting case (not modeled)
                current = 0.0

            # Update ring source current for field computation
            self._ring.current = current

            # Fields and force
            ring_field = self._ring_field_at(position)                # B at magnet pos (from ring)
            magnetic_force = self._magnetic_force_on_magnet(position) # m * ∂z Bz (from ring)

            # Dynamics: +z upward, gravity downward
            acceleration = magnetic_force / self.magnet_params.mass - self.config.gravity

            # Record
            records.append(
                {
                    "time": time,
                    "position": position,
                    "velocity": velocity,
                    "acceleration": acceleration,
                    "flux": flux_current,
                    "emf": emf,
                    "current": current,
                    "ring_field": ring_field,
                    "magnetic_force": magnetic_force,
                    "L": L,
                    "R": R,
                }
            )

            # Integrate kinematics
            flux_last = flux_current
            velocity += acceleration * dt
            position += velocity * dt
            time += dt

        return records

    # ---- Helpers ----

    def _compute_flux(self, magnet_height: float) -> float:
        """Approximate the magnetic flux through the ring area by radial annuli."""
        # Place magnet
        self._magnet.position = (0.0, 0.0, magnet_height)

        # Radial integration (axisymmetric): Φ = ∫ Bz(r,0,0) * 2π r dr
        n = self.ring_params.radial_integration_points
        radii = np.linspace(0.0, self.ring_params.radius, n + 1)
        radial_widths = radii[1:] - radii[:-1]
        sample_radii = 0.5 * (radii[:-1] + radii[1:])

        sensors = np.column_stack(
            (
                sample_radii,
                np.zeros_like(sample_radii),
                np.zeros_like(sample_radii),  # z=0 plane of ring
            )
        )

        b_field = magpy.getB(self._magnet, sensors)
        b_z = b_field[:, 2]
        area_elements = 2.0 * np.pi * sample_radii * radial_widths
        flux = float(np.sum(b_z * area_elements))
        return flux

    def _ring_field_at(self, magnet_height: float) -> Tuple[float, float, float]:
        """Magnetic field created by the induced current at the magnet location."""
        observation = np.array([[0.0, 0.0, magnet_height]])
        b_raw = np.asarray(magpy.getB(self._ring, observation), dtype=float)
        b_vec = np.atleast_2d(b_raw)[0]
        return float(b_vec[0]), float(b_vec[1]), float(b_vec[2])

    def _magnetic_force_on_magnet(self, magnet_height: float) -> float:
        """Estimate axial force on the magnet from the ring's induced field: Fz ≈ m * ∂z Bz."""
        dz = self.config.gradient_step
        if dz <= 0:
            return 0.0
        observation = np.array(
            [[0.0, 0.0, magnet_height + dz], [0.0, 0.0, magnet_height - dz]]
        )
        b_values = magpy.getB(self._ring, observation)[:, 2]  # Bz at z±dz
        grad_bz = (b_values[0] - b_values[1]) / (2.0 * dz)
        force = self._dipole_moment * grad_bz
        return float(force)


# ----------------------------
# Convenience function
# ----------------------------

def simulate_falling_magnet(
    magnet_params: MagnetParams | None = None,
    ring_params: RingParams | None = None,
    config: SimulationConfig | None = None,
) -> List[Dict[str, float | Tuple[float, float, float]]]:
    """Convenience wrapper around :class:`FallingMagnetSimulation`."""
    simulation = FallingMagnetSimulation(
        magnet_params=magnet_params,
        ring_params=ring_params,
        config=config,
    )
    return simulation.run()


# ----------------------------
# Script entry point
# ----------------------------

if __name__ == "__main__":
    data = simulate_falling_magnet()
    df = pd.DataFrame(data)
    # Add ring field column for easier plotting
    df["ring_field2"] = df["ring_field"].apply(lambda v: v[2])  # Bz component

    cols = [
        ("position", "Position (m)", "Magnet Position"),
        ("velocity", "Velocity (m/s)", "Magnet Velocity"),
        ("acceleration", "Acceleration (m/s²)", "Magnet Acceleration"),
        ("magnetic_force", "Force (N)", "Magnetic Force"),
        ("emf", "EMF (V)", "Induced EMF"),
        ("current", "Current (A)", "Induced Current"),
        ("flux", "Flux (T·m²)", "Magnetic Flux"),
        ("ring_field2", "Bz (T)", "Magnetic Field of the Ring at Magnet Position"),
    ]

    fig, axes = plt.subplots(nrows=len(cols), ncols=1, sharex=True, figsize=(8, 22))
    t = df["time"].to_numpy()

    for ax, (col, ylabel, title) in zip(axes, cols):
        ax.plot(t, df[col].to_numpy() if col != "ring_field" else df[col].to_numpy()[:, 2])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Falling Magnet: Kinematics & Electromagnetics", y=0.995)
    fig.tight_layout()
    plt.show()

