def addobject1(bunchdt, data, commdct, key, **kwargs):
    """add an object to the eplus model"""
    obj = newrawobject(data, commdct, key)
    abunch = obj2bunch(data, commdct, obj)
    data.dt[key].append(obj)
    bunchdt[key].append(abunch)
    # adict = getnamedargs(*args, **kwargs)
    for kkey, value in iteritems(kwargs):
        abunch[kkey] = value
    return abunch