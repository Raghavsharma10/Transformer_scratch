def s2tc(s,base=25):
    """Converts seconds to timecode"""
    try:
        f = int(s*base)
    except:
        return "--:--:--:--"
    hh  = int((f / base) / 3600)
    hhd = int((hh % 24))
    mm  = int(((f / base) / 60) - (hh*60))
    ss  = int((f/base) - (hh*3600) - (mm*60))
    ff  = int(f - (hh*3600*base) - (mm*60*base) - (ss*base))
    return "{:02d}:{:02d}:{:02d}:{:02d}".format(hhd, mm, ss, ff)