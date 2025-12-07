from multi_ring import MultiRingSimulation, MagnetParams, TubeParams, ElectricParams, SimConfig, plot_records

def main():
    # ------------------------------
    #  Configuración de la prueba
    # ------------------------------

    # Imán
    magnet_params = MagnetParams(
        radius=0.02,
        height=0.02,
        magnetization=1000,
        mass=0.05
    )

    # Tubo discretizado en anillos
    tube_params = TubeParams(
        ring_radius=0.05,
        ring_count=7,          # ← petición del usuario
        ring_length=0.15,      # longitud total del tubo
        radial_integration_points=200,
        wire_radius=1e-3
    )

    # Parte eléctrica
    electric_params = ElectricParams(
        resistance_per_ring=1.44e-3,   # Ohmios
        include_mutual_inductance=False
    )

    # Configuración de simulación
    sim_config = SimConfig(
        dt=1e-4,
        total_time=0.3,
        initial_height=0.15,
        initial_velocity=0.0,
        gravity=9.81,
        gradient_step=1e-4
    )

    # ------------------------------
    #  Ejecutar simulación
    # ------------------------------

    sim = MultiRingSimulation(
        magnet=magnet_params,
        tube=tube_params,
        electric=electric_params,
        config=sim_config
    )

    print("Ejecutando simulación con anillos...")

    records = sim.run()

    # ------------------------------
    #  Mostrar resultados
    # ------------------------------
    plot_records(records)

    print("Simulación finalizada correctamente.")


if __name__ == "__main__":
    main()