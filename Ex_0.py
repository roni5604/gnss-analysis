import sys, os, csv
import simplekml
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import navpy
from gnssutils import EphemerisManager
import matplotlib.pyplot as plt

def main():

    # Setup directory paths for data and scripts
    # Get path to sample file in data directory, which is located in the parent directory of this notebook
    android_fixes = []
    input_filepath = os.path.join(parent_directory, 'data', 'sample', 'Driving', 'gnss_log_2024_04_13_19_53_33.txt')
    measurements = []

    # Read GNSS data from CSV file
    with open(input_filepath) as csvfile:
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

    # Convert lists to DataFrames for easier manipulation
    android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns=measurements[0])

    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements[
        'Svid']  # Add leading zero to single digit satellite IDs
    measurements.loc[measurements[
                         'ConstellationType'] == '1', 'Constellation'] = 'G'  # Assign constellation to satellite IDs based on constellation type G = GPS (GPS meaning Global Positioning System)
    measurements.loc[measurements[
                         'ConstellationType'] == '3', 'Constellation'] = 'R'  # Assign constellation to satellite IDs based on constellation type R = GLONASS  (GLONASS meaning Russian Global Navigation Satellite System)
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
    # Adding a new column for CH0 based on Cn0DbHz
    measurements['CH0'] = measurements['Cn0DbHz']  # Copying or transforming the C/N0 data
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos'] = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in measurements.columns:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
    else:
        measurements['BiasNanos'] = 0
    if 'TimeOffsetNanos' in measurements.columns:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
    else:
        measurements['TimeOffsetNanos'] = 0

    print(measurements.columns)

    ##########################################################
    ##########################################################
    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (
                measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
    measurements['UnixTime'] = measurements['UnixTime']

    # Split data into measurement epochs
    measurements['Epoch'] = 0
    measurements.loc[
        measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()

    WEEKSEC = 604800
    LIGHTSPEED = 2.99792458e8

    # This should account for rollovers since it uses a week number specific to each measurement

    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (
                measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    # Calculate pseudorange in seconds
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

    # Conver to meters
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']



    manager = EphemerisManager(ephemeris_data_directory)
    #
    epoch = 0
    num_sats = 0
    while num_sats < 5:
         one_epoch = measurements.loc[
             (measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')
         timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
         one_epoch.set_index('SvName', inplace=True)
         num_sats = len(one_epoch.index)
         epoch += 1
    #
    sats = one_epoch.index.unique().tolist()
    ephemeris = manager.get_ephemeris(timestamp, sats)
    # print(timestamp)
    # print(one_epoch[['UnixTime', 'tTxSeconds', 'GpsWeekNumber']])



    sv_position = calculate_satellite_position(ephemeris, one_epoch)
    print(sv_position)
    sv_position.to_csv("parser_to_csv.csv", sep=',')



    # Calculate and plot ECEF coordinates
    ecef_list = []
    for epoch in measurements['Epoch'].unique():
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
        if len(one_epoch.index) > 4:
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(timestamp, sats)
            sv_position = calculate_satellite_position(ephemeris, one_epoch)

            xs = sv_position[['x_k', 'y_k', 'z_k']].to_numpy()
            pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
            pr = pr.to_numpy()

            x = 0
            b = 0
            dp = 0
            x, b, dp = least_squares(xs, pr, x, b)
            ecef_list.append(x)

    ecef_array = np.stack(ecef_list, axis=0)
    lla_array = np.stack(navpy.ecef2lla(ecef_array), axis=1)
    ref_lla = lla_array[0, :]
    ned_array = navpy.ecef2ned(ecef_array, ref_lla[0], ref_lla[1], ref_lla[2])
    #print(sv_position)

    print(ned_array)


    ned_df = pd.DataFrame(ned_array, columns=['N', 'E', 'D'])

######### mission 4 ############
    # Save LLA and NED data to CSV files
    #convert pos x pos y pos z to lan lot alt
    pd.DataFrame(lla_array, columns=['Latitude', 'Longitude', 'Altitude']).to_csv('calculated_position_lan_lot_alt.csv',
                                                                                  index=False)
    pd.DataFrame(ned_array, columns=['N', 'E', 'D']).to_csv('ned_position.csv', index=False)

######### mission 5 ############
    # Create DataFrame with Pos.X, Pos.Y, Pos.Z, Lat, Lon, Alt
    df = pd.DataFrame({
        'Pos.X': ecef_array[:, 0],
        'Pos.Y': ecef_array[:, 1],
        'Pos.Z': ecef_array[:, 2],
        'Latitude': lla_array[:, 0],
        'Longitude': lla_array[:, 1],
        'Altitude': lla_array[:, 2]
    })

    # Save to CSV file
    df.to_csv('position_x_y_z_lan_lot_alt.csv', index=False)

    # Generate KML file
    kml = simplekml.Kml()
    for index, row in df.iterrows():
        kml.newpoint(name=str(index), coords=[(row['Longitude'], row['Latitude'], row['Altitude'])])

    # Save KML file
    kml.save('computed_path.kml')

    print("CSV and KML files have been generated successfully.")


    # Export android fixes DataFrame to CSV
    android_fixes.to_csv('android_position.csv', index=False)

    ###################################################################################################
    ################################ Plot Section #####################################################
    ###################################################################################################
    plt.style.use('dark_background')
    plt.plot(ned_df['E'], ned_df['N'])
    # Add titles
    plt.title('Position Offset From First Epoch')
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()
    columns_for_end_csv = ["GpsTimeNanos", "Svid", "PrM", "Cn0DbHz"]

######### mission 4 ############
def calculate_satellite_position(ephemeris, transmit_time):
    """
    Calculate the position of satellites based on ephemeris data and transmit times.

    Args:
    ephemeris (DataFrame): A DataFrame containing the ephemeris data for each satellite.
    transmit_time (DataFrame): A DataFrame containing transmission times and other relevant data for each satellite.

    Returns:
    DataFrame: A DataFrame containing the computed satellite positions in ECEF coordinates, along with satellite clock corrections.
    """

    # Gravitational constant for Earth (m^3/s^2)
    mu = 3.986005e14
    # Earth's rotation rate (rad/s)
    OmegaDot_e = 7.2921151467e-5
    # Relativity correction coefficient
    F = -4.442807633e-10

    # Initialize DataFrame for storing satellite positions
    sv_position = pd.DataFrame()
    sv_position['sv'] = ephemeris.index
    sv_position.set_index('sv', inplace=True)

    # Time from ephemeris reference epoch
    sv_position['t_k'] = transmit_time['tTxSeconds'] - ephemeris['t_oe']
    A = ephemeris['sqrtA'].pow(2)

    # Compute the mean motion
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']

    # Compute the mean anomaly
    M_k = ephemeris['M_0'] + n * sv_position['t_k']

    # Solve Kepler's equation for the eccentric anomaly
    E_k = M_k
    err = pd.Series(data=[1] * len(sv_position.index))
    i = 0
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1

    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
    delT_oc = transmit_time['tTxSeconds'] - ephemeris['t_oc']
    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris[
        'SVclockDriftRate'] * delT_oc.pow(2)

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))

    Phi_k = v_k + ephemeris['omega']

    # Compute the corrections for argument of latitude, radius and inclination
    sin2Phi_k = np.sin(2 * Phi_k)
    cos2Phi_k = np.cos(2 * Phi_k)
    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

    # Compute the corrected orbital parameters
    u_k = Phi_k + du_k
    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k
    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']

    # Compute positions in the orbital plane
    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)

    Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * ephemeris[
        't_oe']

    sv_position['x_k'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
    sv_position['y_k'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
    sv_position['z_k'] = y_k_prime * np.sin(i_k)
    sv_position['Cn0'] = transmit_time['Cn0DbHz']  # Assuming 'Cn0DbHz' holds C/N0 values


    ################################################################################
    ################################# PSEUDO RANGE #################################
    ################################################################################
    # initial guesses of receiver clock bias and position
    b0 = 0
    x0 = np.array([0, 0, 0])
    xs = sv_position[['x_k', 'y_k', 'z_k']].to_numpy()

    return sv_position

def least_squares(xs, measured_pseudorange, x0, b0):
    """
    Perform a least squares adjustment to estimate the receiver's position and clock bias based on satellite positions and measured pseudoranges.

    Args:
    xs (numpy.ndarray): Satellite positions in ECEF coordinates, shape (n, 3) where n is the number of satellites.
    measured_pseudorange (numpy.ndarray): Observed pseudoranges for each satellite, shape (n,).
    x0 (numpy.ndarray): Initial estimate of receiver position in ECEF coordinates, shape (3,).
    b0 (float): Initial estimate of receiver clock bias in meters.

    Returns:
    tuple: Updated receiver position in ECEF coordinates (numpy.ndarray), updated receiver clock bias in meters (float), and the norm of pseudorange residuals (float).
    """

    dx = 100*np.ones(3)
    b = b0
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    while np.linalg.norm(dx) > 1e-3:
        r = np.linalg.norm(xs - x0, axis=1)
        phat = r + b0
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db
    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp

    print(f"C/N0 values extracted to {output_filepath}")

if __name__ == "__main__":
    main()
