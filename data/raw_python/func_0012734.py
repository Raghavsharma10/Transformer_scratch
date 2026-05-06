def fanpower_bhp(ddtt):
    """return fan power in bhp given the fan IDF object"""
    from eppy.bunch_subclass import BadEPFieldError # here to prevent circular dependency
    try:
        fan_tot_eff = ddtt.Fan_Total_Efficiency # from V+ V8.7.0 onwards
    except BadEPFieldError as e:
        fan_tot_eff = ddtt.Fan_Efficiency 
    pascal = float(ddtt.Pressure_Rise)
    if str(ddtt.Maximum_Flow_Rate).lower() == 'autosize': 
    # str can fail with unicode chars :-(
        return 'autosize'
    else:
        m3s = float(ddtt.Maximum_Flow_Rate)
    return fan_bhp(fan_tot_eff, pascal, m3s)