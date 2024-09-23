import numpy as np
import pandas as pd

# Define the input and output file paths here

input_file = "F:\\DM\\hlist_0.09635.list.list"  # Path to the .list file
output_file = "F:\\DM\\hlist_0.09635.csv"  # Path to save the CSV file

# Function to convert .list to CSV
def convert_to_csv(input_file, output_file, skip_header=1000, max_rows=10000):
    # Load the data using np.genfromtxt
    data = np.genfromtxt(input_file, comments='#', skip_header=skip_header, max_rows=max_rows)
    
    # Define halo properties
    virial_mass = np.log10(data[:, 10])
    virial_radius = np.log10(data[:, 11])
    concentration = np.log10(data[:, 11] / data[:, 12])  # Concentration is virial radius divided by scale length
    velocity_disp = np.log10(data[:, 13])
    vmax = np.log10(data[:, 16])
    spin = np.log10(data[:, 26])
    b_to_a = data[:, 44]
    c_to_a = data[:, 45]
    energy_ratio = data[:, 54]
    peak_mass = np.log10(data[:, 58])
    peak_vmax = np.log10(data[:, 60])
    halfmass_a = data[:, 61]
    peakmass_a = data[:, 67]
    acc_rate = data[:, 64]

    # Create a Pandas DataFrame
    halos = pd.DataFrame({
        'Virial Mass': virial_mass,
        'Virial Radius': virial_radius,
        'Concentration': concentration,
        'Velocity Disp': velocity_disp,
        'Vmax': vmax,
        'Spin': spin,
        'B to A': b_to_a,
        'C to A': c_to_a,
        'Energy ratio': energy_ratio,
        'Peak Mass': peak_mass,
        'peak Vmax': peak_vmax,
        'Halfmass a': halfmass_a,
        'Peakmass a': peakmass_a,
        'Acc Rate': acc_rate
    })

    # Save to CSV
    halos.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

# Call the conversion function
convert_to_csv(input_file, output_file)
