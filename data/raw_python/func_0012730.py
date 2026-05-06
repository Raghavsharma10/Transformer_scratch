def fan_bhp(fan_tot_eff, pascal, m3s):
    """return the fan power in bhp given fan efficiency, Pressure rise (Pa) and flow (m3/s)"""
    # from discussion in
    # http://energy-models.com/forum/baseline-fan-power-calculation
    inh2o = pascal2inh2o(pascal)
    cfm = m3s2cfm(m3s)
    return (cfm * inh2o * 1.0) / (6356.0 * fan_tot_eff)