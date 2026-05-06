def fan_maxcfm(ddtt):
    """return the fan max cfm"""
    if str(ddtt.Maximum_Flow_Rate).lower() == 'autosize':
    # str can fail with unicode chars :-(
        return 'autosize'
    else:
        m3s = float(ddtt.Maximum_Flow_Rate)
        return m3s2cfm(m3s)