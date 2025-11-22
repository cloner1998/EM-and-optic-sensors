"""
Comprehensive examples for EM and Optical sensor modeling.
Demonstrates various real-world scenarios including terrain analysis.
"""

import numpy as np
from dataclass.emDataClass import EmSensorsParameters
from model.emModel import EMSensorsModel
from dataclass.opticDataClass import OpticSensorsParameters
from model.opticModel import OpticalSensorModel
from dataclass.terrain import TerrainProfile
from terrain.terrainAnalyzer import TerrainAnalyzer


def example_1_basic_active_radar():
    """
    Example 1: Basic X-band active radar for maritime surveillance
    Demonstrates monostatic radar equation with moderate parameters
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: X-Band Maritime Surveillance Radar (Active)")
    print("=" * 70)

    params = EmSensorsParameters(
        P_t=1000.0,  # 1 kW transmitter power
        G_t=1000.0,  # 30 dBi antenna gain (linear: 10^(30/10))
        G_r=1000.0,  # Same for receiver
        frequency=10e9,  # 10 GHz (X-band)
        P_r_min=1e-13,  # -130 dBm sensitivity
        L_system=2.0,  # 3 dB system losses (linear: 10^(3/10))
        alpha=0.00005,  # Clear weather attenuation
        D=1.5,  # 1.5 m dish diameter
        SNR_db=20.0,  # 20 dB SNR
        cross_section=10.0,  # 10 m² RCS (small vessel)
        sensor_type="Active"
    )

    model = EMSensorsModel(params)
    R_max = model.effective_range()
    angle_error = model.angle_error()

    print(f"Transmitter Power: {params.P_t} W")
    print(f"Frequency: {params.frequency / 1e9:.1f} GHz")
    print(f"Wavelength: {params.wavelength * 100:.2f} cm")
    print(f"Antenna Gain: {10 * np.log10(params.G_t):.1f} dBi")
    print(f"Target RCS: {params.cross_section} m²")
    print(f"\nResults:")
    print(f"  Maximum Detection Range: {R_max / 1000:.2f} km")
    print(f"  Angular Error: {angle_error * 1000:.3f} milliradians")
    print(f"  Position Error at max range: {R_max * angle_error:.1f} m")


def example_2_passive_em_link():
    """
    Example 2: Passive EM receiver (satellite communication downlink)
    Demonstrates link budget for one-way communication
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Satellite Communication Downlink (Passive)")
    print("=" * 70)

    params = EmSensorsParameters(
        P_t=50.0,  # 50 W satellite transmitter
        G_t=3162.0,  # 35 dBi satellite antenna (10^(35/10))
        G_r=10000.0,  # 40 dBi ground station (10^(40/10))
        frequency=12e9,  # 12 GHz (Ku-band)
        P_r_min=1e-14,  # -140 dBm receiver sensitivity
        L_system=1.5,  # System losses
        alpha=0.00002,  # Low atmospheric attenuation
        D=3.0,  # 3 m ground station dish
        SNR_db=15.0,
        cross_section=1.0,  # Not used for passive, default value = 1.0
        sensor_type="Passive"
    )

    model = EMSensorsModel(params)
    R_max = model.effective_range()
    angle_error = model.angle_error()

    # Calculate received power at typical GEO distance
    R_geo = 36000e3  # 36,000 km
    P_r_geo = model.received_power_passive(R_geo)

    print(f"Satellite Power: {params.P_t} W")
    print(f"Frequency: {params.frequency / 1e9:.1f} GHz (Ku-band)")
    print(f"Ground Station Dish: {params.D} m")
    print(f"\nResults:")
    print(f"  Maximum Link Range: {R_max / 1000:.1f} km")
    print(f"  Angular Accuracy: {angle_error * 1000:.3f} mrad")


def example_4_optical_clear_air():
    """
    Example 4: High-resolution optical imaging sensor in clear conditions
    Demonstrates diffraction-limited performance
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: High-Resolution Optical Imaging Sensor")
    print("=" * 70)

    params = OpticSensorsParameters(
        I_0=1000.0,  # 1000 W/m² (bright daylight)
        I_min=0.1,  # 0.1 W/m² detection threshold
        wavelength=550e-9,  # 550 nm (green light)
        beta=0.001,  # Very clear air (1 km⁻¹)
        D=0.2,  # 20 cm aperture (telescope)
        pixel_pitch=3.5e-6,  # 3.5 μm pixels (typical CMOS)
        focal_length=1.0,  # 1 m focal length
        SNR_db=35.0,  # High SNR
        spot_diameter=10e-6,  # Not used for imaging
        sigma_pos_x=1e-4,  # Not used for imaging
        sensor_type="imaging"
    )

    model = OpticalSensorModel(params)
    R_max = model.effective_range()
    sigma_diff, sigma_smpl, sigma_tot = model.angle_error_imaging()

    print(f"Sensor Parameters:")
    print(f"  Aperture: {params.D * 100:.0f} cm")
    print(f"  Focal Length: {params.focal_length * 100:.0f} cm")
    print(f"  Pixel Size: {params.pixel_pitch * 1e6:.1f} μm")
    print(f"  Wavelength: {params.wavelength * 1e9:.0f} nm")
    print(f"\nResults:")
    print(f"  Effective Range: {R_max / 1000:.2f} km")
    print(f"  Diffraction-Limited Error: {sigma_diff * 1e6:.3f} μrad")
    print(f"  Sampling-Limited Error: {sigma_smpl * 1e6:.3f} μrad")
    print(f"  Total Angular Error: {sigma_tot * 1e6:.3f} μrad")


def example_5_lidar_through_fog():
    """
    Example 5: LIDAR sensor operating in foggy conditions
    Demonstrates high attenuation effects on range
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: LIDAR Sensor in Fog (Non-Imaging PSD)")
    print("=" * 70)

    # Clear conditions
    params_clear = OpticSensorsParameters(
        I_0=5000.0,  # 5000 W/m² (laser intensity)
        I_min=1.0,  # 1 W/m² threshold
        wavelength=905e-9,  # 905 nm (NIR LIDAR)
        beta=0.01,  # Light haze (10 km⁻¹)
        D=0.05,  # 5 cm lens
        pixel_pitch=5e-6,
        focal_length=0.08,  # 8 cm focal length
        SNR_db=25.0,
        spot_diameter=2e-3,  # 2 mm spot
        sigma_pos_x=0.0005,  # 0.5 mm PSD resolution
        sensor_type="non-imaging"
    )

    # Fog conditions
    params_fog = OpticSensorsParameters(
        I_0=5000.0,
        I_min=1.0,
        wavelength=905e-9,
        beta=0.5,  # Dense fog (500 km⁻¹)
        D=0.05,
        pixel_pitch=5e-6,
        focal_length=0.08,
        SNR_db=25.0,
        spot_diameter=2e-3,
        sigma_pos_x=0.0005,
        sensor_type="non-imaging"
    )

    model_clear = OpticalSensorModel(params_clear)
    model_fog = OpticalSensorModel(params_fog)

    R_clear = model_clear.effective_range()
    R_fog = model_fog.effective_range()
    angle_clear = model_clear.angle_error()
    angle_fog = model_fog.angle_error()

    print(f"LIDAR Parameters:")
    print(f"  Wavelength: {params_clear.wavelength * 1e9:.0f} nm (NIR)")
    print(f"  Source Intensity: {params_clear.I_0} W/m²")
    print(f"  PSD Resolution: {params_clear.sigma_pos_x * 1000:.2f} mm")
    print(f"\nClear Conditions (Light Haze):")
    print(f"  Extinction Coeff: {params_clear.beta} km⁻¹")
    print(f"  Effective Range: {R_clear:.1f} m")
    print(f"  Angular Error: {np.degrees(angle_clear):.3f}°")
    print(f"\nFog Conditions:")
    print(f"  Extinction Coeff: {params_fog.beta} km⁻¹")
    print(f"  Effective Range: {R_fog:.1f} m")
    print(f"  Angular Error: {np.degrees(angle_fog):.3f}°")
    print(f"\nRange Reduction Factor: {R_clear / R_fog:.1f}x")


def example_6_terrain_line_of_sight():
    """
    Example 6: Terrain analysis for radio link with obstacle
    Demonstrates LOS clearance calculation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Terrain Line-of-Sight Analysis")
    print("=" * 70)

    # Create terrain profile: flat -> hill -> flat
    num_points = 200
    total_distance = 10000  # 10 km
    x_positions = np.linspace(0, total_distance, num_points)

    # Generate terrain: Gaussian hill in the middle
    hill_center = total_distance / 2
    hill_width = 1000
    hill_height = 80
    h_terrain = hill_height * np.exp(-((x_positions - hill_center) / hill_width) ** 2)

    # Add some noise
    h_terrain += np.random.normal(0, 2, num_points)
    h_terrain = np.maximum(h_terrain, 0)  # Ensure non-negative

    # Sensor configuration
    h_transmitter = 50.0  # 50 m tower
    h_receiver = 30.0  # 30 m tower

    terrain = TerrainProfile(
        x_positions=x_positions,
        h_terrain=h_terrain,
        h_transmitter=h_transmitter,
        h_receiver=h_receiver
    )

    analyzer = TerrainAnalyzer(terrain)

    # Check LOS at various ranges
    test_ranges = [3000,4100, 5000, 7000, 10000]

    print(f"Terrain Configuration:")
    print(f"  Total Distance: {total_distance / 1000:.1f} km")
    print(f"  Transmitter Height: {h_transmitter} m")
    print(f"  Receiver Height: {h_receiver} m")
    print(f"\nLOS Clearance Analysis:")

    for R in test_ranges:
        is_clear, min_clearance = analyzer.check_los_clearance(R)
        status = "CLEAR" if is_clear else "BLOCKED"
        print(f"  Range {R / 1000:.1f} km: {status}")

    # Find maximum clear range
    R_max_clear = analyzer.find_max_clear_range(total_distance)
    print(f"\nMaximum Clear Range: {R_max_clear / 1000:.2f} km")

    # Minimum elevation angle
    theta_min = analyzer.minimum_elevation_angle()
    print(f"Minimum Elevation Angle: {np.degrees(theta_min):.2f}°")


def run_all_examples():
    """Run all examples sequentially"""
    examples = [
        example_1_basic_active_radar,
        example_2_passive_em_link,
        example_4_optical_clear_air,
        example_5_lidar_through_fog,
        example_6_terrain_line_of_sight,
    ]

    print("\n" + "#" * 70)
    print("# EM AND OPTICAL SENSOR MODELING - COMPREHENSIVE EXAMPLES")
    print("#" * 70)

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in example {i}: {str(e)}")

    print("\n" + "#" * 70)
    print("# ALL EXAMPLES COMPLETED")
    print("#" * 70)

if __name__ == "__main__":
    run_all_examples()