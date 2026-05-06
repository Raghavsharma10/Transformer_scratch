def mkUTC(year, month, day, hour, min, sec):
    "similar to python's mktime but for utc"
    spec = [year, month, day, hour, min, sec] + [0, 0, 0]
    utc = time.mktime(spec) - time.timezone
    return utc