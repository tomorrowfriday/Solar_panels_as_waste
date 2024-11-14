import pandas as pd
import os
import glob
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# File path to the source CSV file
file_path = '/Users/tomorrowfriday/Documents/BD/Our_W_in_D/Energy/energy-data/owid-energy-data.csv'

# Define the directory paths for output
current_dir = os.path.dirname(os.path.abspath(__file__))              # Directory where the script is located
output_folder = os.path.join(current_dir, 'data')                     # Folder where output will be saved
output_file = os.path.join(output_folder, 'germany_energy_data.csv')  # CSV file for filtered data
readme_file = os.path.join(output_folder, 'README.txt')               # ReadMe file to log first non-zero solar electricity
# Define the directory path
directory_path = '/Users/tomorrowfriday/Documents/BD/smard_de/'

# Define # of production years
EOF_time=30

# Define file patterns for Actual and Installed generation files
actual_pattern = os.path.join(directory_path, 'Actual_generation_*_Year.csv')
installed_pattern = os.path.join(directory_path, 'Installed_generation_capacity_*_Year.csv')

# Function to load and aggregate data for "Actual" files
def load_actual_data(file_pattern):
    actual_files = glob.glob(file_pattern)
    if not actual_files:
        print("No 'Actual' files found with pattern:", file_pattern)
    actual_df_list = []
    for file in actual_files:
        try:
            df = pd.read_csv(file, delimiter=';')
            if 'Photovoltaics [MWh] Calculated resolutions' in df.columns:
                df['year'] = pd.to_datetime(df['Start date']).dt.year
                df_actual = df[['year', 'Photovoltaics [MWh] Calculated resolutions']].copy()

                df_actual['Photovoltaics [MWh] Calculated resolutions'] = pd.to_numeric(
                    df_actual['Photovoltaics [MWh] Calculated resolutions'].str.replace(',', ''), errors='coerce'
                ) / 1_000_000  # Convert MWh to TWh
                df_actual.rename(columns={'Photovoltaics [MWh] Calculated resolutions': 'Photovoltaics [TWh] Calculated resolutions'}, inplace=True)
                actual_df_list.append(df_actual)
            else:
                print(f"Column 'Photovoltaics [MWh] Calculated resolutions' not found in {file}")
        except pd.errors.ParserError as e:
            print(f"Error reading {file}: {e}")
    if actual_df_list:
        actual_df = pd.concat(actual_df_list)
        actual_df = actual_df.groupby('year').mean().reset_index()
        return actual_df
    else:
        return pd.DataFrame(columns=['year', 'Photovoltaics [TWh] Calculated resolutions'])

# Function to load and aggregate data for "Installed" files
def load_installed_data(file_pattern):
    installed_files = glob.glob(file_pattern)
    if not installed_files:
        print("No 'Installed' files found with pattern:", file_pattern)
    installed_df_list = []
    for file in installed_files:
        try:
            df = pd.read_csv(file, delimiter=';')
            if 'Photovoltaics [MW] Original resolutions' in df.columns:
                df['year'] = pd.to_datetime(df['Start date']).dt.year
                df_installed = df[['year', 'Photovoltaics [MW] Original resolutions']].copy()
                df_installed['Photovoltaics [MW] Original resolutions'] = pd.to_numeric(
                    df_installed['Photovoltaics [MW] Original resolutions'].str.replace(',', ''), errors='coerce'
                )
                installed_df_list.append(df_installed)
            else:
                print(f"Column 'Photovoltaics [MW] Original resolutions' not found in {file}")
        except pd.errors.ParserError as e:
            print(f"Error reading {file}: {e}")
    if installed_df_list:
        installed_df = pd.concat(installed_df_list)
        installed_df = installed_df.groupby('year').mean().reset_index()
        return installed_df
    else:
        return pd.DataFrame(columns=['year', 'Photovoltaics [MW] Original resolutions'])

def load_and_filter_data(file_path):
    """Load CSV data and filter for Germany, ensuring precision in 'solar_electricity' values."""
    # Load CSV with high precision
    df = pd.read_csv(file_path, float_precision='high')
    
    # Filter for Germany and keep only the necessary columns
    germany_df = df[df['country'] == 'Germany'].copy()
    columns_to_keep = ['country', 'year', 'solar_electricity']
    
    # Ensure 'solar_electricity' retains meaningful precision
    germany_df['solar_electricity'] = germany_df['solar_electricity'].apply(lambda x: round(x, 6))
    
    return germany_df[columns_to_keep]

def forecast_solar_electricity(germany_df, forecast_steps=EOF_time):
    """Forecast future solar electricity values for the next EOF_time years."""
    train_df = germany_df.dropna(subset=['solar_electricity']).copy()
    
    train_df['year'] = pd.to_datetime(train_df['year'], format='%Y')
    train_df.set_index('year', inplace=True)
    train_df = train_df.asfreq('AS')  # Set frequency to Annual Start

    model = ExponentialSmoothing(train_df['solar_electricity'], trend="add", seasonal=None)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_years = pd.date_range(start=train_df.index[-1] + pd.DateOffset(years=1), periods=forecast_steps, freq='AS')
    
    forecast_df = pd.DataFrame({
        'country': 'Germany',
        'year': forecast_years,
        'solar_electricity': forecast.values
    })
    
    germany_df.reset_index(drop=True, inplace=True)
    forecast_df['year'] = forecast_df['year'].dt.year  # Convert back to year format
    return pd.concat([germany_df, forecast_df], ignore_index=True)

def add_shifted_column(germany_df, shift_periods=EOF_time):
    """Create a 'solar_waste' column by shifting 'Photovoltaics [TWh] Calculated resolutions' by a specified number of years."""
    germany_df['solar_waste'] = germany_df['Photovoltaics [TWh] Calculated resolutions'].shift(shift_periods)
    return germany_df

def find_first_non_zero_entry(germany_df):
    """Find the first non-zero entry for solar electricity."""
    first_non_zero_index = germany_df[germany_df['solar_electricity'] > 0].index.min()
    if pd.notna(first_non_zero_index):
        year = germany_df.loc[first_non_zero_index, 'year']
        value = germany_df.loc[first_non_zero_index, 'solar_electricity']
    else:
        year, value = "No non-zero values found", "N/A"
    return year, value

def save_to_csv(germany_df, output_file):
    """Save the final DataFrame to a CSV file with numbers formatted as numbers."""
    os.makedirs(output_folder, exist_ok=True)
    
    germany_df['solar_electricity'] = pd.to_numeric(germany_df['solar_electricity'], errors='coerce')
    germany_df['solar_waste'] = pd.to_numeric(germany_df['solar_waste'], errors='coerce')
    
    germany_df.to_csv(output_file, index=False, float_format='%.2f')

def save_readme(year, value, readme_file):
    """Save the first non-zero solar electricity entry to a README file."""
    with open(readme_file, 'w') as f:
        f.write(f"First non-zero solar electricity value:\n")
        f.write(f"Year: {year}\n")
        f.write(f"Solar Electricity: {value} TWh\n")

def estimate_photovoltaics(df):
    """
    Estimates the missing values for 'Photovoltaics [TWh] Calculated resolutions' based on
    the correlation factor with 'solar_electricity', and then estimates missing values for 
    'Photovoltaics [MW] Original resolutions' based on 'Photovoltaics [TWh] Calculated resolutions'.
    After estimating, ensures all values are saved as numbers (floats).

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns 'year', 'solar_electricity', 
                        'Photovoltaics [TWh] Calculated resolutions', and 
                        'Photovoltaics [MW] Original resolutions'.
    
    Returns:
    pd.DataFrame: The input DataFrame with the missing values in both columns filled and saved as numbers.
    """
    
    # Step 1: Estimate missing values in 'Photovoltaics [TWh] Calculated resolutions' based on 'solar_electricity'
    valid_data = df.dropna(subset=['solar_electricity', 'Photovoltaics [TWh] Calculated resolutions'])
    
    # Calculate the correlation factor for available years (using .copy() to avoid SettingWithCopyWarning)
    valid_data = valid_data.copy()
    valid_data['correlation_factor'] = valid_data['Photovoltaics [TWh] Calculated resolutions'] / valid_data['solar_electricity']
    
    # Calculate the average correlation factor
    average_correlation_factor = valid_data['correlation_factor'].mean()
    
    # Estimate missing values for 'Photovoltaics [TWh] Calculated resolutions'
    df['Photovoltaics [TWh] Calculated resolutions'] = df.apply(
        lambda row: row['solar_electricity'] * average_correlation_factor if pd.isna(row['Photovoltaics [TWh] Calculated resolutions']) else row['Photovoltaics [TWh] Calculated resolutions'],
        axis=1
    )
    
    # Step 2: Estimate missing values in 'Photovoltaics [MW] Original resolutions' based on 'Photovoltaics [TWh] Calculated resolutions'
    # Drop rows where either 'Photovoltaics [TWh] Calculated resolutions' or 'Photovoltaics [MW] Original resolutions' is missing
    valid_data = df.dropna(subset=['Photovoltaics [MW] Original resolutions', 'Photovoltaics [TWh] Calculated resolutions'])
    
    # Calculate the correlation factor for available years (using .copy() to avoid SettingWithCopyWarning)
    valid_data = valid_data.copy()
    valid_data['mw_correlation_factor'] = valid_data['Photovoltaics [MW] Original resolutions'] / valid_data['Photovoltaics [TWh] Calculated resolutions']
    
    # Calculate the average correlation factor
    average_mw_correlation_factor = valid_data['mw_correlation_factor'].mean()
    
    # Estimate missing values for 'Photovoltaics [MW] Original resolutions'
    df['Photovoltaics [MW] Original resolutions'] = df.apply(
        lambda row: row['Photovoltaics [TWh] Calculated resolutions'] * average_mw_correlation_factor if pd.isna(row['Photovoltaics [MW] Original resolutions']) else row['Photovoltaics [MW] Original resolutions'],
        axis=1
    )
    
    # Ensure that all relevant columns are saved as numeric (float) values
    df['Photovoltaics [TWh] Calculated resolutions'] = pd.to_numeric(df['Photovoltaics [TWh] Calculated resolutions'], errors='coerce')
    df['Photovoltaics [MW] Original resolutions'] = pd.to_numeric(df['Photovoltaics [MW] Original resolutions'], errors='coerce')
    df['solar_electricity'] = pd.to_numeric(df['solar_electricity'], errors='coerce')

    return df

def calculate_waste_mass(germany_df):
    """
    Calculate 'waste_mass' from 'solar_waste' by dividing 'solar_waste' by 3.33 kWh
    (converted to TWh) and then multiplying by 11.6. Update germany_df with this column.

    Parameters:
    germany_df (pd.DataFrame): DataFrame with 'solar_waste' column in TWh.

    Returns:
    pd.DataFrame: Updated DataFrame with new 'waste_mass' column.
    """
    # Convert 3.33 kWh to TWh
    kWh_to_TWh = 3.33 / 1_000_000  # TWh per 3.33 kWh

    # Calculate waste mass
    germany_df['waste_mass'] = (germany_df['solar_waste'] / kWh_to_TWh) * 11.6

    return germany_df

# Main script execution
def main():
    # Load and aggregate data for Actual and Installed files
    actual_df = load_actual_data(actual_pattern)
    installed_df = load_installed_data(installed_pattern)
    #
    germany_df = load_and_filter_data(file_path)
    germany_df = forecast_solar_electricity(germany_df)

    # Merge the actual and installed data based on the year
    germany_df = germany_df.merge(actual_df[['year', 'Photovoltaics [TWh] Calculated resolutions']], on='year', how='left')
    germany_df = germany_df.merge(installed_df[['year', 'Photovoltaics [MW] Original resolutions']], on='year', how='left')
    
    germany_df = estimate_photovoltaics(germany_df)
    germany_df = add_shifted_column(germany_df)
    # Calculate waste mass and update the DataFrame
    germany_df = calculate_waste_mass(germany_df)

    first_non_zero_year, first_non_zero_value = find_first_non_zero_entry(germany_df)
    
    save_to_csv(germany_df, output_file)
    save_readme(first_non_zero_year, first_non_zero_value, readme_file)
    
    print("Data for Germany has been saved to", output_file)
    print(f"ReadMe.txt has been created with the first non-zero solar electricity information.")

if __name__ == "__main__":
    main()