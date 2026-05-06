def addfunctions(abunch):
    """add functions to epbunch"""

    key = abunch.obj[0].upper()

    #-----------------
    # TODO : alternate strategy to avoid listing the objkeys in snames
    # check if epbunch has field "Zone_Name" or "Building_Surface_Name"
    # and is in group u'Thermal Zones and Surfaces'
    # then it is likely to be a surface.
    # of course we need to recode for surfaces that do not have coordinates :-(
    # or we can filter those out since they do not have
    # the field "Number_of_Vertices"
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
        abunch.__functions.update(func_dict)

    #-----------------
    # print(abunch.getfieldidd )
    names = [
        "CONSTRUCTION",
        "MATERIAL",
        "MATERIAL:AIRGAP",
        "MATERIAL:INFRAREDTRANSPARENT",
        "MATERIAL:NOMASS",
        "MATERIAL:ROOFVEGETATION",
        "WINDOWMATERIAL:BLIND",
        "WINDOWMATERIAL:GLAZING",
        "WINDOWMATERIAL:GLAZING:REFRACTIONEXTINCTIONMETHOD",
        "WINDOWMATERIAL:GAP",
        "WINDOWMATERIAL:GAS",
        "WINDOWMATERIAL:GASMIXTURE",
        "WINDOWMATERIAL:GLAZINGGROUP:THERMOCHROMIC",
        "WINDOWMATERIAL:SCREEN",
        "WINDOWMATERIAL:SHADE",
        "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
              ]
    if key in names:
        func_dict = {
            'rvalue': fh.rvalue,
            'ufactor': fh.ufactor,
            'rvalue_ip': fh.rvalue_ip,  # quick fix for Santosh. Needs to thought thru
            'ufactor_ip': fh.ufactor_ip,  # quick fix for Santosh. Needs to thought thru
            'heatcapacity': fh.heatcapacity,
        }
        abunch.__functions.update(func_dict)

    names = [
        'FAN:CONSTANTVOLUME',
        'FAN:VARIABLEVOLUME',
        'FAN:ONOFF',
        'FAN:ZONEEXHAUST',
        'FANPERFORMANCE:NIGHTVENTILATION',
              ]
    if key in names:
        func_dict = {
            'f_fanpower_bhp': fh.fanpower_bhp,
            'f_fanpower_watts': fh.fanpower_watts,
            'f_fan_maxcfm': fh.fan_maxcfm,
        }
        abunch.__functions.update(func_dict)
    # =====
    # code for references
    #-----------------
    # add function zonesurfaces
    if key == 'ZONE':
        func_dict = {'zonesurfaces':fh.zonesurfaces}
        abunch.__functions.update(func_dict)

    #-----------------
    # add function subsurfaces
    # going to cheat here a bit
    # check if epbunch has field "Zone_Name"
    # and is in group u'Thermal Zones and Surfaces'
    # then it is likely to be a surface attached to a zone
    fields = abunch.fieldnames
    try:
        group = abunch.getfieldidd('key')['group']
    except KeyError as e:  # some pytests don't have group
        group = None
    if group == u'Thermal Zones and Surfaces':
        if "Zone_Name" in fields:
            func_dict = {'subsurfaces':fh.subsurfaces}
            abunch.__functions.update(func_dict)

    return abunch