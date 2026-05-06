def create_sphere(rows=10, cols=10, depth=10, radius=1.0, offset=True,
                  subdivisions=3, method='latitude'):
    """Create a sphere

    Parameters
    ----------
    rows : int
        Number of rows (for method='latitude' and 'cube').
    cols : int
        Number of columns (for method='latitude' and 'cube').
    depth : int
        Number of depth segments (for method='cube').
    radius : float
        Sphere radius.
    offset : bool
        Rotate each row by half a column (for method='latitude').
    subdivisions : int
        Number of subdivisions to perform (for method='ico')
    method : str
        Method for generating sphere. Accepts 'latitude' for latitude-
        longitude, 'ico' for icosahedron, and 'cube' for cube based
        tessellation.

    Returns
    -------
    sphere : MeshData
        Vertices and faces computed for a spherical surface.
    """
    if method == 'latitude':
        return _latitude(rows, cols, radius, offset)
    elif method == 'ico':
        return _ico(radius, subdivisions)
    elif method == 'cube':
        return _cube(rows, cols, depth, radius)
    else:
        raise Exception("Invalid method. Accepts: 'latitude', 'ico', 'cube'")