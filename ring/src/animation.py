"""
src/animation.py

Animación de la simulación multi-anillo.

Requisitos: matplotlib, numpy, magpylib (para la simulación)
Asume que `MultiRingSimulation` devuelve una lista de dicts (records) con claves:
 - 'time', 'position', 'velocity', ...
 - 'current' (o 'currents', 'I', o similar) : array-like de tamaño N (corrientes por anillo)
 - 'flux' (opcional): array-like de tamaño N

Este script detecta automáticamente la clave de corrientes y genera una animación.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from multi_ring import MultiRingSimulation, MagnetParams, TubeParams, ElectricParams, SimConfig, plot_records
import warnings

def detect_current_key(records):
    """
    Detecta la clave más plausible que contiene las corrientes por anillo.
    Preferencias: 'current', 'currents', 'I' (exactas). Si no las encuentra,
    toma la primera clave cuyo valor sea array-like y de longitud > 1.
    Retorna (key_name, ndarray_of_shape_nt_N).
    """
    if not records:
        raise ValueError("La lista 'records' está vacía.")
    first = records[0]
    keys = list(first.keys())

    # prefered names
    preferred = ["current", "currents", "I", "curr", "I_vec"]
    for p in preferred:
        if p in first:
            arr = np.atleast_1d(np.asarray([np.atleast_1d(rec[p]) for rec in records]))
            return p, arr

    # otherwise search for any array-like key with length>1
    for k in keys:
        v0 = first[k]
        if isinstance(v0, (list, tuple, np.ndarray)):
            # check that it's vector-like and consistent across records
            try:
                arr = np.atleast_1d(np.asarray([np.atleast_1d(rec[k]) for rec in records]))
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[1] > 1:
                return k, arr

    raise KeyError("No se encontró clave con corrientes por anillo en los registros.")

def animate_simulation(records, tube_params, save=False, filename="animation.gif", fps=30):
    """
    records: lista de dicts devueltos por MultiRingSimulation.run()
    tube_params: instancia TubeParams (contiene ring_count, ring_length, ring_radius)
    save: si True guarda la animación (gif o mp4 dependiendo de extensión)
    """

    # --- extraer tiempos y posiciones ---
    times = np.array([rec["time"] for rec in records])
    positions = np.array([rec["position"] for rec in records])

    # detectar corrientes por anillo
    try:
        cur_key, currents = detect_current_key(records)
        print(f"[INFO] → Corrientes detectadas en la clave: '{cur_key}'")
    except KeyError as e:
        print("[WARN] No se detectó clave de corrientes por anillo: ", e)
        raise

    # asegurar shape (nt, N)
    currents = np.asarray(currents)
    if currents.ndim == 1:
        # caso improbable: cada registro tenía una sola corriente. Convertir a (nt,1)
        currents = currents.reshape(-1, 1)

    nt, N = currents.shape
    # seguridad: si len(times) != nt, advertir
    if nt != len(times):
        warnings.warn(f"Numero de tiempos ({len(times)}) distinto de filas de 'currents' ({nt}). Ajustando nt = min(...).")
        nt = min(nt, len(times))
        times = times[:nt]
        positions = positions[:nt]
        currents = currents[:nt, :]

    # Construir posiciones axiales de los anillos (coherente con MultiRingSimulation)
    ring_count = int(tube_params.ring_count)
    Ltot = float(tube_params.ring_length)
    # distribuir anillos de -L/2 a +L/2 en el eje z (igual que la simulacion)
    ring_z = np.linspace(-Ltot/2, Ltot/2, ring_count)

    # límites gráficos
    z_min = min(np.min(ring_z) - 0.05, np.min(positions) - 0.05)
    z_max = max(np.max(ring_z) + 0.05, np.max(positions) + 0.05)

    # preparar figura
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.set_xlim(-1.2 * tube_params.ring_radius, 1.2 * tube_params.ring_radius)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel("X (m) — eje radial (solo ilustrativo)")
    ax.set_ylabel("Z (m) — eje axial")
    ax.set_title("Animación: imán cayendo a través de anillos (corriente codificada por color)")

    # representar el imán como un rectángulo (square marker)
    magnet_marker, = ax.plot([0.0], [positions[0]], marker="s", color="darkred", markersize=12, label="Imán")

    # crear líneas para cada anillo
    ring_lines = []
    for zi in ring_z:
        ln, = ax.plot([], [], lw=6, solid_capstyle="round")
        ring_lines.append(ln)

    # escala de color: use diverging colormap centered at 0
    cmap = plt.cm.coolwarm
    absmax = np.max(np.abs(currents)) if np.max(np.abs(currents)) != 0 else 1.0

    # texto indicativos
    txt_time = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    txt_current = ax.text(0.02, 0.90, "", transform=ax.transAxes)

    # función de actualización
    def update(frame):
        # actualiza magneto
        z = positions[frame]
        magnet_marker.set_data([0.0], [z])

        # actualizar color de anillos según corriente en ese frame
        Iframe = currents[frame]  # shape (N,)
        # normalizar a [-1,1]
        if absmax > 0:
            norms = Iframe / absmax
        else:
            norms = Iframe
        colors = [cmap((val + 1.0) / 2.0) for val in norms]

        for i, (ln, zi) in enumerate(zip(ring_lines, ring_z)):
            # dibujar línea horizontal representando el anillo
            x0, x1 = -tube_params.ring_radius, tube_params.ring_radius
            ln.set_data([x0, x1], [zi, zi])
            ln.set_color(colors[i])
            ln.set_linewidth(6)

        txt_time.set_text(f"t = {times[frame]:.3f} s")
        # mostrar corriente máxima (valor absoluto) y signo
        imax = np.max(np.abs(Iframe))
        txt_current.set_text(f"I_max = {imax:.3e} A")

        return [magnet_marker, *ring_lines, txt_time, txt_current]

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 * (times[1] - times[0]) if len(times) > 1 else 50, blit=True)

    plt.legend(loc="upper right")

    if save:
        fname = filename.lower()
        if fname.endswith(".gif"):
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer)
            print(f"[INFO] Animación guardada en {filename}")
        elif fname.endswith(".mp4"):
            try:
                writer = FFMpegWriter(fps=fps)
                anim.save(filename, writer=writer)
                print(f"[INFO] Animación guardada en {filename}")
            except Exception as e:
                warnings.warn("No se pudo guardar MP4 (comprueba ffmpeg). Error: " + str(e))
        else:
            # intentar gif si no reconoce extensión
            try:
                writer = PillowWriter(fps=fps)
                anim.save(filename + ".gif", writer=writer)
                print(f"[INFO] Animación guardada en {filename}.gif")
            except Exception as e:
                warnings.warn("No se pudo guardar la animación: " + str(e))

    plt.show()
    return anim


# ---------------------------
# PRUEBA DIRECTA DESDE SCRIPT
# ---------------------------
if __name__ == "__main__":
    # Parámetros de ejemplo (3 anillos)
    magnet_params = MagnetParams(radius=0.02, height=0.02, magnetization=1000, mass=0.05)
    tube_params = TubeParams(ring_radius=0.05, ring_count=7, ring_length=0.15, radial_integration_points=120)
    electric_params = ElectricParams(resistance_per_ring=1.44e-3, include_mutual_inductance=False)
    sim_config = SimConfig(dt=5e-4, total_time=0.5, initial_height=0.6, initial_velocity=0.5, gradient_step=1e-4)

    print("[INFO] Ejecutando simulación (esto puede tardar según parámetros)...")
    sim = MultiRingSimulation(magnet=magnet_params, tube=tube_params, electric=electric_params, config=sim_config)
    results = sim.run()

    # debug: mostrar claves del primer registro
    print("[DEBUG] Claves del primer registro:", results[0].keys())

    # call animation (guarda GIF por defecto)
    animate_simulation(results, tube_params, save=True, filename="multi_ring_animation.gif", fps=20)
