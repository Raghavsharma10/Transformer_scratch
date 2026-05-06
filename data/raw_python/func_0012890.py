def getadistus(data, commdct):
    """docstring for fname"""
    objkey = "ZoneHVAC:AirDistributionUnit".upper()
    objindex = data.dtls.index(objkey)
    objcomm = commdct[objindex]
    adistutypefield = "Air Terminal Object Type"
    ifield = getfieldindex(data, commdct, objkey, adistutypefield)
    adistus = objcomm[ifield]['key']
    return adistus