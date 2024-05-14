
# GNSS Data Analysis Script

## Overview
This Python script is designed for in-depth analysis of GNSS (Global Navigation Satellite System) data. It processes raw GNSS log files to extract and calculate satellite positions, receiver positions, and other navigation solutions using GPS data and ephemeris information. The goal is to facilitate the understanding and visualization of GNSS data characteristics for research or practical applications.
- A more detailed explanation including articles on the subject and helpful materials See this [link](https://www.johnsonmitchelld.com/2021/03/14/least-squares-gps.html).

## Prerequisites
- **Python 3.8+**: This version or newer is required for script compatibility.
- **Libraries**: Ensure that Pandas, NumPy, Matplotlib, and NavPy are installed for data manipulation and visualization.

## Installation

### Python Installation
Download and install Python from the official site: [python.org](https://www.python.org/).

### Dependency Installation
Install the required Python libraries using pip:
```bash
pip install pandas numpy matplotlib navpy
```

### Directory Structure
The script expects a structured directory with a `data` folder in the parent directory of this script containing necessary GNSS log files and ephemeris data.

## Usage
To run the script, navigate to the script's directory in the command line and execute:
```bash
python Ex0_finel.py
```

### Important Note on File Paths
In the script, you need to set the correct path to your data file. Replace the placeholder path with your actual file path:
```python
# Replace 'your_path_here' with the actual path to your GNSS log file
input_filepath = os.path.join(parent_directory,'gnss-analysis', 'data', 'sample', 'Driving', 'gnss_log_2024_04_13_19_53_33.txt')
```

## Task Explanation
This project involves analyzing GNSS data using Python. The primary goal is to understand GNSS data and implement algorithms for GNSS analysis.

- For detailed task description and instructions, please refer to this [Google Docs document](https://docs.google.com/document/d/1DDLrA2BoJ4RKa4ahbm2prtseBdgM-2C9UbHO-JwSasw/edit?usp=sharing).

Our work is based on and adapted from another project aimed at understanding GNSS analysis and algorithms. For more information, see this [link](https://www.johnsonmitchelld.com/2021/03/14/least-squares-gps.html).

## Script Functionality and Detailed Code Explanation

### Part 1: Data Preparation
**Path Setup**: The script starts by configuring the directories and file paths required to manage data inputs and outputs efficiently. This setup ensures that the script can access GNSS data files and ephemeris information stored in a structured directory hierarchy.
```python
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
```

### Part 2: Data Reading
**File Reading**: It then reads the GNSS log files, selectively ignoring any comments and extracting relevant data into lists.
```python
with open(input_filepath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0][0] == '#':  # Skip comments
            if 'Fix' in row[0]:  # Check if the row contains the header for the fix data
                android_fixes = [row[1:]]
            elif 'Raw' in row[0]:
                measurements = [row[1:]]
        else:
            if row[0] == 'Fix':
                android_fixes.append(row[1:])
            elif row[0] == 'Raw':
                measurements.append(row[1:])
```

### Part 3: Data Conversion
**DataFrame Conversion**: Converts the extracted lists of data into Pandas DataFrames to facilitate easier data manipulation.
```python
android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
measurements = pd.DataFrame(measurements[1:], columns=measurements[0])
```

### Part 4: Calculations
**Formatting and Filtering**: The script formats satellite IDs and filters out non-GPS data.
```python
measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
measurements = measurements.loc[measurements['Constellation'] == 'G']
```

**GNSS Time Conversion**: Converts GNSS time to Unix time.
```python
measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
```

**Pseudorange Calculation**: Calculates pseudoranges based on satellite and receiver time.
```python
measurements['tRxGnssNanos'] = (measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - 
                                (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0]))
measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']
measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']
```

### Part 5: Satellite Position Calculation
**Satellite Position Calculation**: Calculates the position of satellites based on ephemeris data and transmit times.
```python
sv_position = calculate_satellite_position(ephemeris, one_epoch)
```
The `calculate_satellite_position` function computes the position of each satellite in ECEF coordinates using the Keplerian elements provided in the ephemeris data.

### Part 6: Least Squares Estimation
**Least Squares Estimation**: Estimates the receiver's position and clock bias using the least squares method.
```python
x, b, dp = least_squares(xs, pr, x, b)
```
The `least_squares` function performs iterative adjustments to minimize the difference between observed and calculated pseudoranges, resulting in an estimate of the receiver's position.

### Part 7: Output Generation
**CSV and KML Exports**: Exports the computed positions to CSV and KML files for further analysis and visualization.
```python
android_fixes.to_csv('android_position.csv', index=False)
plt.plot(ned_df['E'], ned_df['N'])
plt.title('Position Offset From First Epoch')
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.gca().set_aspect('equal', adjustable='box')
```
The script saves various positional data in different formats and generates a KML file for visualization in mapping software.

## Contributing
Contributions to improve the script are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
