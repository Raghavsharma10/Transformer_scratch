def fan_watts(fan_tot_eff, pascal, m3s):
    """return the fan power in watts given fan efficiency, Pressure rise (Pa) and flow (m3/s)"""
    # got this from a google search
    bhp = fan_bhp(fan_tot_eff, pascal, m3s)
    return bhp2watts(bhp)