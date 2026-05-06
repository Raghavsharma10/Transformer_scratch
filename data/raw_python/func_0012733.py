def watts2pascal(watts, cfm, fan_tot_eff):
    """convert and return inputs for E+ in pascal and m3/s"""
    bhp = watts2bhp(watts)
    return bhp2pascal(bhp, cfm, fan_tot_eff)