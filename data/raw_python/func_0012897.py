def makebunches(data, commdct):
    """make bunches with data"""
    bunchdt = {}
    ddtt, dtls = data.dt, data.dtls
    for obj_i, key in enumerate(dtls):
        key = key.upper()
        bunchdt[key] = []
        objs = ddtt[key]
        for obj in objs:
            bobj = makeabunch(commdct, obj, obj_i)
            bunchdt[key].append(bobj)
    return bunchdt