def sample(raster, coords):
    """Sample a raster at given coordinates

    Given a list of coordinates, return a list of x,y,z triples with z coordinates sampled from an input raster

    Parameters:
        raster (rasterio): raster dataset to sample
        coords: array of tuples containing coordinate pairs (x,y) or triples (x,y,z)

    Returns:
        result: array of tuples containing coordinate triples (x,y,z)
        
    """
    if len(coords[0]) == 3:
        logging.info('Input is a 3D geometry, z coordinate will be updated.')
        z = raster.sample([(x, y) for x, y, z in coords], indexes=raster.indexes)
    else:
        z = raster.sample(coords, indexes=raster.indexes)

    result = [(vert[0], vert[1], vert_z) for vert, vert_z in zip(coords, z)]

    return result