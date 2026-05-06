def plantloopfieldlists(data):
    """return the plantloopfield list"""
    objkey = 'plantloop'.upper()
    numobjects = len(data.dt[objkey])
    return [[
        'Name',
        'Plant Side Inlet Node Name',
        'Plant Side Outlet Node Name',
        'Plant Side Branch List Name',
        'Demand Side Inlet Node Name',
        'Demand Side Outlet Node Name',
        'Demand Side Branch List Name']] * numobjects