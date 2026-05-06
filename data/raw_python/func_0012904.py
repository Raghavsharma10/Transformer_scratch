def addfunctions2new(abunch, key):
    """add functions to a new bunch/munch object"""
    snames = [
        "BuildingSurface:Detailed",
        "Wall:Detailed",
        "RoofCeiling:Detailed",
        "Floor:Detailed",
        "FenestrationSurface:Detailed",
        "Shading:Site:Detailed",
        "Shading:Building:Detailed",
        "Shading:Zone:Detailed", ]
    snames = [sname.upper() for sname in snames]
    if key in snames:
        func_dict = {
            'area': fh.area,
            'height': fh.height,  # not working correctly
            'width': fh.width,  # not working correctly
            'azimuth': fh.azimuth,
            'tilt': fh.tilt,
            'coords': fh.getcoords,  # needed for debugging
        }
        try:
            abunch.__functions.update(func_dict)
        except KeyError as e:
            abunch.__functions = func_dict
    return abunch