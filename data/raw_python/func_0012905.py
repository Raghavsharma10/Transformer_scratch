def idfreader(fname, iddfile, conv=True):
    """read idf file and return bunches"""
    data, commdct, idd_index = readidf.readdatacommdct(fname, iddfile=iddfile)
    if conv:
        convertallfields(data, commdct)
    # fill gaps in idd
    ddtt, dtls = data.dt, data.dtls
    # skiplist = ["TABLE:MULTIVARIABLELOOKUP"]
    nofirstfields = iddgaps.missingkeys_standard(
        commdct, dtls,
        skiplist=["TABLE:MULTIVARIABLELOOKUP"])
    iddgaps.missingkeys_nonstandard(None, commdct, dtls, nofirstfields)
    bunchdt = makebunches(data, commdct)
    return bunchdt, data, commdct, idd_index