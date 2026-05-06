def buildingname(ddtt):
    """return building name"""
    idf = ddtt.theidf
    building = idf.idfobjects['building'.upper()][0]
    return building.Name