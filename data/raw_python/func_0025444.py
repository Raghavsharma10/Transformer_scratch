def extract_storm_objects(label_grid, data, x_grid, y_grid, times, dx=1, dt=1, obj_buffer=0):
    """
    After storms are labeled, this method extracts the storm objects from the grid and places them into STObjects.
    The STObjects contain intensity, location, and shape information about each storm at each timestep.

    Args:
        label_grid: 2D or 3D array output by label_storm_objects.
        data: 2D or 3D array used as input to label_storm_objects.
        x_grid: 2D array of x-coordinate data, preferably on a uniform spatial grid with units of length.
        y_grid: 2D array of y-coordinate data.
        times: List or array of time values, preferably as integers
        dx: grid spacing in same units as x_grid and y_grid.
        dt: period elapsed between times
        obj_buffer: number of extra pixels beyond bounding box of object to store in each STObject

    Returns:
        storm_objects: list of lists containing STObjects identified at each time.
    """
    storm_objects = []
    if len(label_grid.shape) == 3:
        ij_grid = np.indices(label_grid.shape[1:])
        for t, time in enumerate(times):
            storm_objects.append([])
            object_slices = list(find_objects(label_grid[t], label_grid[t].max()))
            if len(object_slices) > 0:
                for o, obj_slice in enumerate(object_slices):
                    if obj_buffer > 0:
                        obj_slice_buff = [slice(np.maximum(0, osl.start - obj_buffer),
                                                np.minimum(osl.stop + obj_buffer, label_grid.shape[l + 1]))
                                          for l, osl in enumerate(obj_slice)]
                    else:
                        obj_slice_buff = obj_slice
                    storm_objects[-1].append(STObject(data[t][obj_slice_buff],
                                                      np.where(label_grid[t][obj_slice_buff] == o + 1, 1, 0),
                                                      x_grid[obj_slice_buff],
                                                      y_grid[obj_slice_buff],
                                                      ij_grid[0][obj_slice_buff],
                                                      ij_grid[1][obj_slice_buff],
                                                      time,
                                                      time,
                                                      dx=dx,
                                                      step=dt))
                    if t > 0:
                        dims = storm_objects[-1][-1].timesteps[0].shape
                        storm_objects[-1][-1].estimate_motion(time, data[t - 1], dims[1], dims[0])
    else:
        ij_grid = np.indices(label_grid.shape)
        storm_objects.append([])
        object_slices = list(find_objects(label_grid, label_grid.max()))
        if len(object_slices) > 0:
            for o, obj_slice in enumerate(object_slices):
                if obj_buffer > 0:
                    obj_slice_buff = [slice(np.maximum(0, osl.start - obj_buffer),
                                            np.minimum(osl.stop + obj_buffer, label_grid.shape[l + 1]))
                                      for l, osl in enumerate(obj_slice)]
                else:
                    obj_slice_buff = obj_slice
                storm_objects[-1].append(STObject(data[obj_slice_buff],
                                                  np.where(label_grid[obj_slice_buff] == o + 1, 1, 0),
                                                  x_grid[obj_slice_buff],
                                                  y_grid[obj_slice_buff],
                                                  ij_grid[0][obj_slice_buff],
                                                  ij_grid[1][obj_slice_buff],
                                                  times,
                                                  times,
                                                  dx=dx,
                                                  step=dt))
    return storm_objects