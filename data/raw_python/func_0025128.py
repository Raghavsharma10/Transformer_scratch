def load_map_coordinates(map_file):
    """
    Loads map coordinates from netCDF or pickle file created by util.makeMapGrids.

    Args:
        map_file: Filename for the file containing coordinate information.

    Returns:
        Latitude and longitude grids as numpy arrays.
    """
    if map_file[-4:] == ".pkl":
        map_data = pickle.load(open(map_file))
        lon = map_data['lon']
        lat = map_data['lat']
    else:
        map_data = Dataset(map_file)
        if "lon" in map_data.variables.keys():
            lon = map_data.variables['lon'][:]
            lat = map_data.variables['lat'][:]
        else:
            lon = map_data.variables["XLONG"][0]
            lat = map_data.variables["XLAT"][0]
    return lon, lat