import numpy as np
import matplotlib.pyplot as plt
from dataclass.em_data_class import EmSensorsParameters
from model.em_model import EMSensorsModel
from dataclass.optic_data_class import OpticSensorsParameters
from model.optic_model import OpticalSensorModel
from dataclass.terrain import TerrainProfile
from terrain.terrain_analyzer import TerrainAnalyzer


def plot_range_vs_frequency():
    """Visualization: How frequency affects range"""
    frequencies = np.logspace(8, 11, 50)  # 100 MHz to 100 GHz
    ranges = []

    for freq in frequencies:
        params = EmSensorsParameters(
            P_t=100, G_t=100, G_r=100,
            frequency=freq, P_r_min=1e-12,
            L_system=1.0, alpha=0.0001,
            D=1.0, SNR_db=20, cross_section=5.0,
            sensor_type="Active"
        )
        model = EMSensorsModel(params)
        ranges.append(model.effective_range())

    plt.figure(figsize=(10, 6))
    plt.semilogx(frequencies / 1e9, np.array(ranges) / 1000)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Maximum Range (km)')
    plt.title('EM Sensor: Range vs Frequency')
    plt.grid(True)
    plt.savefig('plots/range_vs_frequency.png', dpi=300)
    plt.show()


def plot_terrain_profile():
    """Visualization: Terrain with LOS"""
    # Create terrain
    x = np.linspace(0, 10000, 200)
    h_terrain = 50 * np.exp(-((x - 5000) / 1000) ** 2)

    terrain = TerrainProfile(x, h_terrain,
                             h_transmitter=60, h_receiver=40)
    analyzer = TerrainAnalyzer(terrain)

    h_LOS = analyzer.line_of_sight(x, 10000)

    plt.figure(figsize=(12, 6))
    plt.plot(x / 1000, h_terrain, 'brown', linewidth=2, label='Terrain')
    plt.plot(x / 1000, h_LOS, 'b--', linewidth=2, label='Line of Sight')
    plt.fill_between(x / 1000, 0, h_terrain, color='brown', alpha=0.3)
    plt.axhline(terrain.h_transmitter, color='g', linestyle=':', label='Transmitter')
    plt.axhline(terrain.h_receiver, color='r', linestyle=':', label='Receiver')
    plt.xlabel('Distance (km)')
    plt.ylabel('Height (m)')
    plt.title('Terrain Profile and Line-of-Sight Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/terrain_los.png', dpi=300)
    plt.show()


def plot_angle_error_vs_snr():
    """Visualization: Angular accuracy vs SNR"""
    snr_db_range = np.linspace(0, 40, 50)
    em_errors = []
    optical_errors = []

    for snr in snr_db_range:
        # EM sensor
        em_params = EmSensorsParameters(
            P_t=100, G_t=100, G_r=100, frequency=10e9,
            P_r_min=1e-12, L_system=1.0, alpha=0,
            D=1.0, SNR_db=snr, cross_section=5.0,
            sensor_type="Active"
        )
        em_model = EMSensorsModel(em_params)
        em_errors.append(em_model.angle_error() * 1000)  # mrad

        # Optical sensor
        opt_params = OpticSensorsParameters(
            I_0=1000, I_min=1, wavelength=550e-9,
            beta=0.01, D=0.1, SNR_db=snr,
            sensor_type="imaging"
        )
        opt_model = OpticalSensorModel(opt_params)
        optical_errors.append(opt_model.angle_error() * 1e6)  # μrad

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogy(snr_db_range, em_errors)
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Angular Error (milliradians)')
    ax1.set_title('EM Sensor: Angle Error vs SNR')
    ax1.grid(True)

    ax2.semilogy(snr_db_range, optical_errors)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Angular Error (microradians)')
    ax2.set_title('Optical Sensor: Angle Error vs SNR')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('plots/angle_error_vs_snr.png', dpi=300)
    plt.show()


def plot_atmospheric_effects():
    """Visualization: Attenuation effects on range"""
    beta_values = np.logspace(-3, 0, 50)  # 0.001 to 1.0
    optical_ranges = []

    for beta in beta_values:
        params = OpticSensorsParameters(
            I_0=1000, I_min=0.1,
            wavelength=550e-9, beta=beta,
            D=0.1, SNR_db=30,
            sensor_type="imaging"
        )
        model = OpticalSensorModel(params)
        optical_ranges.append(model.effective_range())

    plt.figure(figsize=(10, 6))
    plt.loglog(beta_values, optical_ranges)
    plt.xlabel('Extinction Coefficient β (1/m)')
    plt.ylabel('Effective Range (m)')
    plt.title('Optical Sensor: Range vs Atmospheric Attenuation')
    plt.grid(True, which="both")

    # Add condition labels
    plt.axvline(0.001, color='g', linestyle='--', alpha=0.5, label='Clear Air')
    plt.axvline(0.01, color='y', linestyle='--', alpha=0.5, label='Light Haze')
    plt.axvline(0.1, color='orange', linestyle='--', alpha=0.5, label='Fog')
    plt.axvline(0.5, color='r', linestyle='--', alpha=0.5, label='Dense Fog')

    plt.legend()
    plt.savefig('plots/atmospheric_effects.png', dpi=300)
    plt.show()


def run_all_plots():
    plots = [
        plot_range_vs_frequency,
        plot_terrain_profile,
        plot_atmospheric_effects,
    ]

    print("\n" + "#" * 70)
    print("# EM AND OPTICAL SENSOR MODELING - COMPREHENSIVE PLOTS")
    print("#" * 70)

    for i, plot in enumerate(plots, 1):
        try:
            plot()
        except Exception as e:
            print(f"\n❌ Error in example {i}: {str(e)}")

    print("\n" + "#" * 70)
    print("# ALL PRINTS COMPLETED")
    print("#" * 70)

if __name__ == '__main__':
    run_all_plots()