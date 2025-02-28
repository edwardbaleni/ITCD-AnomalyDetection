# %%
import pandas as pd


# Define the path to the Excel file
file_path = 'Spectral.xlsx'

# Read the Excel file
sheets = pd.read_excel(file_path, sheet_name=None)

# %%

# Create a dictionary to hold the dataframes
dataframes = {}

# Iterate over each sheet and store the dataframe in the dictionary
for sheet_name, df in sheets.items():
    if df.shape[1] == 2:  # Ensure the sheet has exactly two columns
        dataframes[sheet_name] = df



# %%
# Label the columns of each dataframe
for sheet_name, df in dataframes.items():
    df.columns = ['x', 'y']

    # Multiply all x-axis values by 1000
    for sheet_name, df in dataframes.items():
        df['x'] = df['x'] * 1000
# %%
import matplotlib.pyplot as plt
# Plot the data from each dataframe and save the plot
plt.figure(figsize=(18, 9))
for sheet_name, df in dataframes.items():
    if 'red' in sheet_name.lower():
        plt.plot(df['x'], df['y'], 'r', label=sheet_name, linewidth=2.5)
    elif 'green' in sheet_name.lower():
        plt.plot(df['x'], df['y'], 'g', label=sheet_name, linewidth=2.5)
    elif 'blue' in sheet_name.lower():
        plt.plot(df['x'], df['y'], 'b', label=sheet_name, linewidth=2.5)
# plt.xlabel('Wavelength (µm)', fontsize=14)
plt.ylabel('Reflectance', fontsize=14)
# plt.title('Spectral Data')
plt.legend()
plt.savefig('spectral_data_plot.png')
plt.show()

# Plot the data for 'reg' and 'nir' on the same plot and save the plot
plt.figure(figsize=(18, 9))
for sheet_name, df in dataframes.items():
    if 'reg' in sheet_name.lower():
        plt.plot(df['x'], df['y'], color='purple', label=sheet_name, linewidth=2.5)
    elif 'nir' in sheet_name.lower():
        plt.plot(df['x'], df['y'], color='orange', label=sheet_name, linewidth=2.5)
# plt.xlabel('Wavelength (µm)', fontsize=14)
plt.ylabel('Reflectance', fontsize=14)
# plt.title('Reg and NIR Data')
plt.legend()
plt.savefig('reg_nir_data_plot.png')
plt.show()
# %%
# Bending energy