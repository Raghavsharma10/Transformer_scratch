def addfunctions(dtls, bunchdt):
    """add functions to the objects"""
    snames = [
        "BuildingSurface:Detailed",
        "Wall:Detailed",
        "RoofCeiling:Detailed",
        "Floor:Detailed",
        "FenestrationSurface:Detailed",
        "Shading:Site:Detailed",
        "Shading:Building:Detailed",
        "Shading:Zone:Detailed", ]
    for sname in snames:
        if sname.upper() in bunchdt:
            surfaces = bunchdt[sname.upper()]
            for surface in surfaces:
                func_dict = {
                    'area': fh.area,
                    'height': fh.height,  # not working correctly
                    'width': fh.width,  # not working correctly
                    'azimuth': fh.azimuth,
                    'tilt': fh.tilt,
                    'coords': fh.getcoords,  # needed for debugging
                }
                try:
                    surface.__functions.update(func_dict)
                except KeyError as e:
                    surface.__functions = func_dict