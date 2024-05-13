
# GNSS Data Analysis Script

## Overview
This Python script is designed for in-depth analysis of GNSS (Global Navigation Satellite System) data. It processes raw GNSS log files to extract and calculate satellite positions, receiver positions, and other navigation solutions using GPS data and ephemeris information. The goal is to facilitate the understanding and visualization of GNSS data characteristics for research or practical applications.
- A more detailed explanation including articles on the subject and helpful materials See this https://www.johnsonmitchelld.com/2021/03/14/least-squares-gps.html

## Prerequisites
- **Python 3.7+**: This version or newer is required for script compatibility.
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
python init.py
```

## Script Functionality and Detailed Code Explanation

### Data Preparation
**Path Setup**: The script starts by configuring the directories and file paths required to manage data inputs and outputs efficiently. This setup ensures that the script can access GNSS data files and ephemeris information stored in a structured directory hierarchy.
```python
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
```

### Data Reading
**File Reading**: It then reads the GNSS log files, selectively ignoring any comments and extracting relevant GNSS data based on tags like 'Fix' and 'Raw'. This selective reading helps in focusing only on useful data, skipping metadata or other non-essential information.
```python
with open(input_filepath) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0][0] == '#': continue  # Skips comment lines
        ...
```

### Data Processing
**DataFrame Conversion**: Converts the extracted lists of data into Pandas DataFrames to facilitate easier data manipulation. Using DataFrames allows the script to utilize powerful data manipulation and analysis functions provided by Pandas.
```python
android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
measurements = pd.DataFrame(measurements[1:], columns=measurements[0])
```

**Filtering and Formatting**: This step filters out all non-GPS data and formats satellite IDs to ensure consistent processing across different data entries. This standardization is crucial for accurate GNSS analysis.
```python
measurements = measurements.loc[measurements['Constellation'] == 'G']
```

### Calculations
**Time and Position Calculations**: The script calculates accurate times and positions using detailed GNSS-specific adjustments like satellite clock biases and Earth's rotation corrections. These calculations are vital for deriving precise navigation solutions.
```python
measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
```

### Output Generation
**CSV Exports**: After processing, the script outputs computed positions into CSV files for further analysis or archiving. It also creates plots to visualize data, providing a graphical representation of the trajectory or position offsets.
```python
android_fixes.to_csv('android_position.csv', index=False)
```

**Plotting**: Visualizes the trajectory or position offsets using Matplotlib. This graphical representation helps in understanding the movement patterns and positional accuracy of the GNSS receiver.
```python
plt.plot(ned_df['E'], ned_df['N'])
plt.title('Position Offset From First Epoch')
plt.show()
```

## Contributing
Contributions to improve the script are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
