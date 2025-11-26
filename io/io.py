import numpy as np
from dataclass.em_data_class import EmSensorsParameters
from model.em_model import EMSensorsModel
from dataclass.optic_data_class import OpticSensorsParameters
from model.optic_model import OpticalSensorModel
import os

def read_config_file(filename):
    """Read configuration from KEY=VALUE format text file"""
    params = {}

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value

        return params
    except FileNotFoundError:
        print(f"❌ Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return None


def analyze_em_from_file(filename):
    """Analyze EM sensor from configuration file"""
    print("\n" + "=" * 70)
    print(f"Reading EM Sensor Configuration: {filename}")
    print("=" * 70)

    params_dict = read_config_file(filename)
    if not params_dict:
        return

    try:
        params = EmSensorsParameters(
            P_t=params_dict.get('P_t', 1000.0),
            G_t=params_dict.get('G_t', 1000.0),
            G_r=params_dict.get('G_r', 1000.0),
            frequency=params_dict.get('frequency', 10e9),
            P_r_min=params_dict.get('P_r_min', 1e-13),
            L_system=params_dict.get('L_system', 2.0),
            alpha=params_dict.get('alpha', 0.00005),
            D=params_dict.get('D', 1.5),
            SNR_db=params_dict.get('SNR_db', 20.0),
            cross_section=params_dict.get('cross_section', 10.0),
            sensor_type=params_dict.get('sensor_type', 'Active')
        )

        model = EMSensorsModel(params)
        R_max = model.effective_range()
        angle_error = model.angle_error()

        print("\n" + "=" * 70)
        print("EM SENSOR ANALYSIS RESULTS")
        print("=" * 70)

        print(f"\nConfiguration:")
        print(f"  Type: {params.sensor_type}")
        print(f"  Transmitter Power: {params.P_t} W")
        print(f"  Frequency: {params.frequency / 1e9:.2f} GHz")
        print(f"  Wavelength: {params.wavelength * 100:.2f} cm")
        print(f"  Antenna Gain: {10 * np.log10(params.G_t):.1f} dBi")
        print(f"  Aperture: {params.D} m")
        print(f"  SNR: {params.SNR_db} dB")
        if params.sensor_type.lower() == 'active':
            print(f"  Target RCS: {params.cross_section} m²")

        print(f"\n{'=' * 70}")
        print("RESULTS:")
        print(f"{'=' * 70}")
        print(f"✓ Maximum Detection Range: {R_max / 1000:.2f} km ({R_max:.1f} m)")
        print(f"✓ Angular Error: {angle_error * 1000:.3f} mrad = {np.degrees(angle_error):.4f}°")
        print(f"✓ Position Error at max range: {R_max * angle_error:.1f} m")

        # Save results
        save_results(filename, R_max, angle_error, 'EM')

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")


def analyze_optical_from_file(filename):
    """Analyze Optical sensor from configuration file"""
    print("\n" + "=" * 70)
    print(f"Reading Optical Sensor Configuration: {filename}")
    print("=" * 70)

    params_dict = read_config_file(filename)
    if not params_dict:
        return

    try:
        params = OpticSensorsParameters(
            I_0=params_dict.get('I_0', 1000.0),
            I_min=params_dict.get('I_min', 0.1),
            wavelength=params_dict.get('wavelength', 550e-9),
            beta=params_dict.get('beta', 0.001),
            D=params_dict.get('D', 0.2),
            pixel_pitch=params_dict.get('pixel_pitch', 3.5e-6),
            focal_length=params_dict.get('focal_length', 1.0),
            SNR_db=params_dict.get('SNR_db', 35.0),
            spot_diameter=params_dict.get('spot_diameter', 1e-3),
            sigma_pos_x=params_dict.get('sigma_pos_x', 1e-4),
            sensor_type=params_dict.get('sensor_type', 'imaging')
        )

        model = OpticalSensorModel(params)
        R_max = model.effective_range()
        angle_error = model.angle_error()

        print("\n" + "=" * 70)
        print("OPTICAL SENSOR ANALYSIS RESULTS")
        print("=" * 70)

        print(f"\nConfiguration:")
        print(f"  Type: {params.sensor_type}")
        print(f"  Wavelength: {params.wavelength * 1e9:.1f} nm")
        print(f"  Aperture: {params.D * 100:.1f} cm")
        print(f"  Attenuation Coefficient: {params.beta} (1/m)")
        print(f"  Focal Length: {params.focal_length * 100:.1f} cm")
        print(f"  SNR: {params.SNR_db} dB")

        print(f"\n{'=' * 70}")
        print("RESULTS:")
        print(f"{'=' * 70}")
        if R_max > 1000:
            print(f"✓ Effective Range: {R_max / 1000:.2f} km ({R_max:.1f} m)")
        else:
            print(f"✓ Effective Range: {R_max:.1f} m")
        print(f"✓ Angular Error: {angle_error * 1e6:.3f} miu_rad = {np.degrees(angle_error):.6f}°")
        if R_max < 1e6:
            print(f"✓ Position Error at max range: {R_max * angle_error:.2f} m")

        # Save results
        save_results(filename, R_max, angle_error, 'Optical')

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")


def save_results(input_filename, range_m, angle_error, sensor_type):
    """Save results to output file in io/output directory"""

    # Create output directory if it doesn't exist
    output_dir = os.path.join('output')
    os.makedirs(output_dir, exist_ok=True)

    # Extract just the filename without path
    base_filename = os.path.basename(input_filename)
    # Replace .txt with _results.txt
    output_filename = str(base_filename.replace('.txt', '_results.txt'))
    # Create full output path
    output_path: str = os.path.join(output_dir, output_filename)

    try:
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"SENSOR ANALYSIS RESULTS\n")
            f.write(f"Input File: {input_filename}\n")
            f.write(f"Sensor Type: {sensor_type}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Effective Range (m): {range_m:.2f}\n")
            f.write(f"Effective Range (km): {range_m / 1000:.2f}\n")
            f.write(f"Angular Error (rad): {angle_error:.6e}\n")

            if sensor_type == 'EM':
                f.write(f"Angular Error (mrad): {angle_error * 1000:.3f}\n")
            else:
                f.write(f"Angular Error (miu_rad): {angle_error * 1e6:.3f}\n")

            f.write(f"Angular Error (degrees): {np.degrees(angle_error):.6f}\n")
            f.write(f"Position Error at max range (m): {range_m * angle_error:.2f}\n")

        print(f"\n✓ Results saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ Error saving results: {str(e)}")

if __name__ == '__main__':
    analyze_em_from_file("input/input_em.txt")
    analyze_optical_from_file("input/input_optics.txt")

