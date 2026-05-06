def label_storm_objects(data, method, min_intensity, max_intensity, min_area=1, max_area=100, max_range=1,
                        increment=1, gaussian_sd=0):
    """
    From a 2D grid or time series of 2D grids, this method labels storm objects with either the Enhanced Watershed
    or Hysteresis methods.

    Args:
        data: the gridded data to be labeled. Should be a 2D numpy array in (y, x) coordinate order or a 3D numpy array
            in (time, y, x) coordinate order
        method: "ew" or "watershed" for Enhanced Watershed or "hyst" for hysteresis
        min_intensity: Minimum intensity threshold for gridpoints contained within any objects
        max_intensity: For watershed, any points above max_intensity are considered as the same value as max intensity.
            For hysteresis, all objects have to contain at least 1 pixel that equals or exceeds this value
        min_area: (default 1) The minimum area of any object in pixels.
        max_area: (default 100) The area threshold in pixels at which the enhanced watershed ends growth. Object area
            may exceed this threshold if the pixels at the last watershed level exceed the object area.
        max_range: Maximum difference between the maximum and minimum value in an enhanced watershed object before
            growth is stopped.
        increment: Discretization increment for the enhanced watershed
        gaussian_sd: Standard deviation of Gaussian filter applied to data
    Returns:
        label_grid: an ndarray with the same shape as data in which each pixel is labeled with a positive integer value.
    """
    if method.lower() in ["ew", "watershed"]:
        labeler = EnhancedWatershed(min_intensity, increment, max_intensity, max_area, max_range)
    else:
        labeler = Hysteresis(min_intensity, max_intensity)
    if len(data.shape) == 2:
        label_grid = labeler.label(gaussian_filter(data, gaussian_sd))
        label_grid[data < min_intensity] = 0
        if min_area > 1:
            label_grid = labeler.size_filter(label_grid, min_area)
    else:
        label_grid = np.zeros(data.shape, dtype=int)
        for t in range(data.shape[0]):
            label_grid[t] = labeler.label(gaussian_filter(data[t], gaussian_sd))
            label_grid[t][data[t] < min_intensity] = 0
            if min_area > 1:
                label_grid[t] = labeler.size_filter(label_grid[t], min_area)
    return label_grid