"""
Rock Joint Sets Analysis Tool

This module processes dip-direction/dip measurements of geological discontinuities,
calculates their centroids using clustering methods, and visualizes the data on a stereonet.

Features:
    - K-means and K-medoids clustering for joint set identification
    - Stereonet visualization of poles and density contours
    - Statistical analysis of clustering results
    - Multiple clustering solution comparison

Units:
    All angular measurements are in degrees.

Note:
    K-means clustering results may vary between runs due to random initialization.
    For more stable results, consider using at first k-medoids.

Dependencies:
    - numpy: For numerical operations and array handling
    - matplotlib.pyplot: For plotting and visualization
    - mplstereonet: For stereonet plotting
    - kmedoids: For k-medoids clustering (alternative to k-means)

Example Usage:
    data_file = 'dirbuz_buz.txt'
    num = 5
    run_example(data_file, num, tittle=None
    
Educational Software
@author: Fernando García Bastante
Universidad de Vigo
"""

import mplstereonet as mpl
import matplotlib.pyplot as plt
import numpy as np
import kmedoids

def read_data(data_file, delimit=""):
    """
    Read dip direction and dip measurements from a file.

    Parameters:
        data_file (str): Path to the input file containing space-separated
                        dip direction and dip values in degrees.
        
    Returns:
        numpy.ndarray: A 2D array where each row contains [dip_direction, dip]
                    measurements in degrees.
    """
    data = np.genfromtxt(data_file, delimiter=delimit)
    return data

def dip_dir_to_strikes(data):
    ''' 
    Converts dip direction/dip values into strike/dip values.
    Parameters:
        data (numpy.ndarray): A 2D array where the first column is dip direction 
                        and the second column is dip.
    Returns:
        tuple: Two numpy arrays, one for strike and one for dip values
    '''
    # reading dip_dirs/dips
    dips_dir, dips = data[:, 0], data[:, 1]
    # dips_dirs to strikes
    strikes = dips_dir - 90
    strikes = np.where(strikes < 0, strikes + 360, strikes)
    return strikes, dips

def draw_data_centroids(data, strike_cent=None, dip_cent=None,
                        dens_cont=True, tittle='poles'):
    '''
    Plots the poles of the dataset and their centroids on a stereonet.
    Parameters:
        strikes (numpy.ndarray): Array of strike values.
        dips (numpy.ndarray): Array of dip values.
        strike_cent (numpy.ndarray, optional): Array of centroid strike values.
        dip_cent (numpy.ndarray, optional): Array of centroid dip values.

    Returns:
        None
    '''
    #strikes, dips = dip_dir_to_strikes(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')
    
    # Plot data
    if data is not None:
        # dip_dirs to strikes(dips)
        strikes, dips = dip_dir_to_strikes(data)
        # plotting
        ax.pole(strikes, dips, ms=1)
        if dens_cont is True:
            ax.density_contour(strikes, dips, measurement='poles')
    
    #ax.pole(strikes, dips, ms=1)
    #ax.density_contour(strikes, dips, measurement='poles')
    # Plot centroids
    if strike_cent is not None:
        ax.pole(strike_cent, dip_cent, 'ro', ms=12)
    
    # title and turn on a grid
    ax.set_title(tittle, pad=30)
    ax.grid(True)
    plt.show()
        
def centroids_cal(data, num):
    '''
    Calculates centroids of the data using k-means clustering.
    Parameters:
        strikes (numpy.ndarray): Array of strike values.
        dips (numpy.ndarray): Array of dip values.
        num (int): Number of clusters.

    Returns:
        tuple: Two numpy arrays containing centroid strike and dip values.
        '''
    strikes, dips = dip_dir_to_strikes(data)
    centroids = mpl.kmeans(strikes, dips, num=num)
    # lon/lat to dir_dip/dip
    strike_cent, dip_cent = mpl.geographic2pole(*zip(*centroids))
    return np.round(strike_cent, 1), np.round(dip_cent,1)
    
def strikes_to_dip_dir(strikes):
    """
    Convert strike values to dip direction values.

    Parameters:
        strikes (numpy.ndarray): Array of strike values in degrees

    Returns:
        numpy.ndarray: Array of dip direction values in degrees
    """
    dip_dir = strikes + 90
    dip_dir = np.where(dip_dir > 360, dip_dir - 360, dip_dir)
    return dip_dir

def strikes_dips_to_centroids(strike_cent, dip_cent):
    """
    Convert centroid strike/dip values to dip_direction/dip values.

    Parameters:
        strike_cent (numpy.ndarray): Array of centroid strike values in degrees
        dip_cent (numpy.ndarray): Array of centroid dip values in degrees

    Returns:
        numpy.ndarray: Array of centroid dip direction values in degrees
    """
    return np.transpose([strikes_to_dip_dir(strike_cent), dip_cent])

def dd_to_ll(data):
    """
    Convert dip direction and dip measurements to longitude/latitude coordinates.

    Parameters:
        data (numpy.ndarray): Array of shape (n, 2) containing [dip_direction, dip]
                            measurements in degrees

    Returns:
        tuple: Two arrays (longitude, latitude) representing pole coordinates
    """
    strikes, dips = dip_dir_to_strikes(data)
    return mpl.stereonet_math.pole(strikes, dips)

def mpl_kmean_n_times(data_file, num, n_times=10, tol=1, plot=True):
    """
    Perform k-means clustering multiple times and return unique solutions.

    Parameters:
        data_file (str): Path to input data file
        num (int): Number of clusters to identify
        n_times (int, optional): Number of times to run k-means.
        tol (float, optional): Minimal difference, in degrees, in order to establish unique solutions

    Returns:
        list: List of unique clustering solutions, each containing strike and dip centroids
    """    
    data = read_data(data_file)

    centroid=[]
    for i in range(n_times):
        centroid.append(centroids_cal(data, num))

    index = unique_index(centroid, tol=tol)
    c_c = [centroid[item] for item in index]

    if plot is True:    
        i = 0
        for centers in c_c:    
            strike_cent = centers[0]
            dip_cent = centers[1]
            title='k_means from mplstereonet, solution ' + str(i)
            draw_data_centroids(data, strike_cent, dip_cent, tittle=title)
            i = i + 1
    return c_c

def k_medoids(data_file, num):
    """
    Find cluster centers using the k-medoids clustering algorithm.

    This function implements the FastPAM algorithm for k-medoids clustering,
    which is more robust to outliers than k-means. It works directly with
    angular distances between poles on a stereonet.

    Parameters:
        data_file (str): Path to the input file containing dip direction/dip
                        measurements in degrees
        num (int): Number of clusters to identify

    Returns:
        numpy.ndarray: Array of shape (num, 2) containing cluster medoids,
                    where each row contains [dip_direction, dip] in degrees

    Notes:
        - Uses the FastPAM algorithm implementation from the kmedoids package
        - Automatically visualizes results on a stereonet plot
        - Angular distances are used as the dissimilarity measure
    """

    data = read_data(data_file)
    points = dd_to_ll(data)
    lon, lat = points

    # get the angular distance
    dist = lambda x: mpl.stereonet_math.angular_distance(x, points)
    distmatrix_sq = np.array([dist(item) for item in np.array(points).T])

    # apply the fasterpam algoritm
    km = kmedoids.KMedoids(num, method='fasterpam')
    c = km.fit(distmatrix_sq)

    # getting the poles from results
    medoids_km = c.medoid_indices_
    c_c = np.array([data[i] for i in medoids_km])
    strikes_c, dips_c = dip_dir_to_strikes(c_c)

    # plotting the results
    draw_data_centroids(data=data,
            strike_cent=strikes_c,
            dip_cent=dips_c,
            tittle='k_medoids')
    return c_c

def kmeans_mod(data, num, pre_centers=None, tol=1, plot=False):
    """
    Find centers of multi-modal clusters using a modified k-means approach for spherical data.

    This function implements a modified version of the k-means algorithm specifically
    adapted for spherical measurements in geological data.

    Parameters:
        data_file (str): Path to the input file containing dip measurements
        num (int): Number of clusters to identify
        pre_centers (numpy.ndarray, optional): Pre-defined initial centers for clustering.
                                        Shape should be (n, 2) where n is the number
        tol (float, optional): Minimal mean difference, in degrees, in order to establish unique solutions                                 of centers.

    Returns:
        tuple: Contains:
            - c_c (tuple): Two arrays of strike and dip values for cluster centers
            - mean_dist (float): Mean angular distance of points to their centers
            - n_clos (numpy.ndarray): Cluster assignments for each data point
    """
    
    # data to lon/lan
    points = dd_to_ll(data)
    lon, lat = points

    dist = lambda x: mpl.stereonet_math.angular_distance(x, points)

    # select the initial centers
    if pre_centers is None:
        #  random initial centers from data
        idx = np.random.choice(len(points[0]), num, replace=False)
        points_ = np.array(points).T
        # array num x 2
        centers = points_[idx]
    else:
        # convert pre_centers to array if necessary
        pre_centers = np.array(dd_to_ll(pre_centers)).T

        # random initial centers from a preselection of centers (v.g. peaks)
        if len(pre_centers) != num:
            idx = np.random.choice(len(pre_centers), num, replace=False)
            centers = pre_centers[idx]
        else:
            centers = pre_centers

    # apply kmeans
    while True:
        dists = np.array([dist(item) for item in centers]).T
        n_clos = dists.argmin(axis=1)
        mean_dist = np.mean(dists[n_clos])

        # kmeans dont find always a solution with num clusters!
        if len(np.unique(n_clos)) != num:
            mean_dist = 1000
            break

        new_centers = []
        for i in range(num):
            mask = n_clos == i
            _, vecs = mpl.cov_eig(lon[mask], lat[mask], bidirectional=True)
            new_centers.append(mpl.stereonet_math.cart2sph(*vecs[:, -1]))

        if np.allclose(centers, new_centers, atol=tol):
            break
        else:
            centers = new_centers

    # lat/lon to strike/dip
    c_c = mpl.geographic2pole(*zip(*new_centers))
    if plot==True:
        draw_data_centroids(data, strike_cent=c_c[0], dip_cent=c_c[1])
    return c_c, mean_dist, n_clos

def unique_index(centroids, tol=1):
    '''
    Filtering of multiple solutions -when using mpl_kmean_n_times- that are
    roughly duplicated
    
    Parameters:
        centroids (numpy.ndarray): Array of cluster centroids
        tol (float, optional): Minimal mean difference, in degrees, in order to 
                                establish unique solutions
    Returns:
        index (set): index of unique solutions
    '''
    c_c_n = np.array(centroids)
    sol_number = len(c_c_n)
    index = []
    strike_mean = []
    dip_mean = []
    for i in range(sol_number):
        strike_mean.append(np.mean(c_c_n[i, 0]))
        dip_mean.append(np.mean(c_c_n[i, 1]))

    # look for the -aprox.- same results
    for i in range(sol_number-1):
        for j in range(i+1, len(strike_mean)):
            if np.allclose(strike_mean[i], strike_mean[j], atol=tol):
                if np.allclose(dip_mean[i], dip_mean[j], atol=tol):
                    index.append(j)

    # just get the index of different results
    index = set(range(sol_number)) - set(index)

    return index

def filter_centroids(data_file, centroids, k_std=3):
    """
    Filter data points that are k standard deviations away from cluster centers.

    Parameters:
        data_file (str): Path to the input file containing space-separated
                        dip direction and dip values in degrees.
        centroids (numpy.ndarray): Array of cluster centroids
        k_std (float, optional): Number of standard deviations for filtering.
                                Defaults to 3.

    Returns:
        tuple: Contains:
            - c_c_filtered (numpy.ndarray): Filtered cluster centroids
            - data_sets (numpy.ndarray): data with cluster assignments for each
                                        one, with 1000 indicating filtered-out points
    """
    data = read_data(data_file)
    # convert to lon/lat
    strikes, dips = dip_dir_to_strikes(data)
    lon, lat = mpl.stereonet_math.pole(strikes, dips)
    points = lon, lat

    strikes_c, dips_c = dip_dir_to_strikes(centroids)
    centroids = np.array(mpl.stereonet_math.pole(strikes_c, dips_c)).T

    # calculate distances
    dist = lambda x: mpl.stereonet_math.angular_distance(x, points)
    dists = np.array([dist(item) for item in centroids]).T

    # each data to to the closer cluster
    n_clos = dists.argmin(axis=1)
    min_dist = np.min(dists, axis=1)

    # data filtering and new centroids
    new_strike_c, new_dip_c = [], []
    for i in range(len(centroids)):
        # mask the i cluster and get theirs dists
        mask_i = n_clos == i
        dist_i = min_dist[mask_i]

        # calculate the limit distance
        lim_dist = np.mean(dist_i) + k_std * np.std(dist_i)

        # mask for data in and out of limit distance
        mask_in = ((n_clos == i) & (min_dist <= lim_dist))
        mask_out = ((n_clos == i) & (min_dist > lim_dist))

        # mark data outs with 1000
        n_clos[mask_out] = 1000

        # get the masked data
        strikes_mask, dips_mask = strikes[mask_in], dips[mask_in]

        # get the poles with the filtered data
        centroids = mpl.kmeans(strikes_mask, dips_mask, num=1)
        new_strike, new_dip = mpl.geographic2pole(*zip(*centroids))
        new_strike_c.append(new_strike)
        new_dip_c.append(new_dip)

    # strikes to dip_dirs
    new_dip_dir_c = [strikes_to_dip_dir(item) for item in new_strike_c]
    c_c_filtered = np.column_stack([new_dip_dir_c, new_dip_c])

    # plotting results
    draw_data_centroids(data[n_clos!=1000],
                        strike_cent=new_strike_c,
                        dip_cent=new_dip_c,
                        tittle='filtered_kmeans'
                        )
    # adding each set to each data (add 1_D to n_clos to append with numpy)
    nclos = np.expand_dims(n_clos, axis=1)
    data_sets = np.append(data, nclos, axis=1)

    return np.round(c_c_filtered, 1), data_sets

def stats_circ(data_i, centroid_i):
    """
    Calculate circular statistics for a cluster of measurements.

    Computes mean and standard deviation for both dip direction and dip,
    accounting for the circular nature of directional data.

    Parameters:
        data_i (numpy.ndarray): Array of measurements for one cluster
        centroid_i (numpy.ndarray): Centroid of the cluster

    Returns:
        tuple: Contains:
            - dip_dirs_mean (float): Mean dip direction
            - dips_mean (float): Mean dip
            - dip_dirs_std (float): Standard deviation of dip directions
            - dips_std (float): Standard deviation of dips
    """
    # unpack data and centroid
    dip_dirs_i, dips_i = data_i.T[0], data_i.T[1]
    dip_dir_c, dip_c = centroid_i.T[0], centroid_i.T[1]

    # calculate mean and std
    diff_dirs = []
    dips = []
    for i in range(len(dip_dirs_i)):
        dis = np.mod(dip_dir_c - dip_dirs_i[i] + 180, 360) - 180
        dis_s = np.mod(dip_dir_c - (dip_dirs_i[i] + 180) + 180, 360) - 180
        if np.abs(dis_s) < np.abs(dis):
            diff_dirs.append(dis_s)
            if dip_c < 45:
                dips.append(-dips_i[i])
            else:
                dips.append(180 - dips_i[i])
        else:
            diff_dirs.append(dis)
            dips.append(dips_i[i])

    dip_dirs_mean = dip_dir_c - np.mean(diff_dirs)
    if dip_dirs_mean >= 360:
        dip_dirs_mean -= 360
    dip_dirs_std = np.std(diff_dirs)

    dips_mean = np.mean(dips)
    dips_std = np.std(dips)

    return dip_dirs_mean, dips_mean, dip_dirs_std, dips_std

def stats(dataset, c_c_filtered):
    """
    Calculate statistics for all clusters in the dataset.

    Parameters:
        data (numpy.ndarray): Dataset of measurements and cluster assigments
        c_c_filtered (tuple): Centroids

    Returns:
        list: List of tuples containing statistics for each cluster
            (dip_dir_mean, dip_mean, dip_dir_std, dip_std)
    """

    # calculate the mean and std of data for each cluster
    stat_centr = []
    for i in range(len(c_c_filtered)):
        # mask for each cluster
        # mask = n_clos == i
        data_set_i = dataset[dataset[:,2]==i]
        data_i = data_set_i[:,0:2]
        stat_centr.append(stats_circ(data_i, c_c_filtered[i]))

    return np.round(stat_centr, 1)

def clusters(data_file, num=4, pre_centers=None, n_times=20, tol=1, plot='sol'):
    """
    Perform multiple k-means clustering runs and analyze solutions.

    This function runs the modified k-means algorithm multiple times to find
    stable clustering solutions, sorts them by mean distance, and optionally
    plots the results.

    Parameters:
        data (str): Path to input data file
        num (int): Number of clusters to identify
        pre_centers (numpy.ndarray, optional): Pre-defined initial centers
        n_times (int, optional): Number of clustering runs. Defaults to 100
        plot (str, optional): Plot mode - 'sol' for all solutions or 'best' for
                            best solution only. Defaults to 'sol'
        tol (float, optional): Minimal mean difference, in degrees, in order to establish unique solutions

    Returns:
        list: List of arrays containing cluster centroids in dip direction/dip format
    """
    data = read_data(data_file)
    # call to kmeans_mod
    c_c = []
    m_d = []
    n_clos = []
    for i in range(n_times):
        center_clusters, mean_dists, closest = kmeans_mod(data,
                                                            num=num,
                                                            pre_centers=pre_centers,
                                                            tol=tol,
                                                            plot=False)
        c_c.append(center_clusters)
        m_d.append(mean_dists)
        n_clos.append(closest)

    # n_times x num strike x num dip array
    c_c, m_d = np.array(c_c), np.array(m_d)
    
    # remove duplicates
    index = unique_index(c_c, tol)
    c_c = [c_c[item] for item in index]
    m_d = [m_d[item] for item in index]
    n_clos = [n_clos[item] for item in index]

    # results sorted by mean distance
    ord_dist = np.argsort(m_d)
    c_c_ord_dist = [c_c[item] for item in ord_dist]

    # plotting and getting strikes/dips
    if plot == 'sol':
        titles = 'possible solution number '
        c_c = c_c_ord_dist
    else:
        titles = 'best kmeans_mod'
        c_c = c_c_ord_dist[0]

    # results in c_c_kmod
    c_c_kmod = []
    for i in range(len(c_c)):
        draw_data_centroids(data, strike_cent=c_c[i][0], dip_cent=c_c[i][1],
                            tittle=titles + str(i))

        # strikes to dir/dip
        c_c_kmod.append((strikes_to_dip_dir(c_c[i][0]), c_c[i][1]))

    c_c_kmod = [np.array(c_c_kmod[i]).T for i in range(len(c_c_kmod))]
    return np.round(c_c_kmod, 1)

def run_example(data_file, num, tittle=None):
    ''' Example function that reads data from data_file, processes it, computes
    num centroids and visualizes the results.
    
    Returns:
        tuple: numpy arrays containing centroid strike and dip values.
    '''
    data = read_data(data_file)
    strike_cent, dip_cent = centroids_cal(data, num)
    dip_dir_cent = strikes_to_dip_dir(strike_cent)
    draw_data_centroids(data, strike_cent, dip_cent, tittle)
    return np.vstack((dip_dir_cent, dip_cent)).T


if __name__ == "__main__":
    data_file = 'my_set.txt'
    num = 3
    tittle = 'EXAMPLE: YOU CAN RUN IT SEVERAL TIMES'
    centroids = run_example(data_file, num)
