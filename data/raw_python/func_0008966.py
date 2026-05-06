def grid_coords_from_corners(upper_left_corner, lower_right_corner, size):
    ''' Points are the outer edges of the UL and LR pixels. Size is rows, columns.
    GC projection type is taken from Points. '''
    assert upper_left_corner.wkt == lower_right_corner.wkt
    geotransform = np.array([upper_left_corner.lon, -(upper_left_corner.lon - lower_right_corner.lon) / float(size[1]), 0,
                            upper_left_corner.lat, 0, -(upper_left_corner.lat - lower_right_corner.lat) / float(size[0])])
    return GridCoordinates(geotransform=geotransform,
                               wkt=upper_left_corner.wkt,
                               y_size=size[0],
                               x_size=size[1])