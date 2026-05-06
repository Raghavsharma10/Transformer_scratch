def add_gtc_header_table_row(t, telpars):
    """
    Add a row with current values to GTC table

    Arguments
    ---------
    t : `~astropy.table.Table`
        The table to append row to
    telpars : list
        list returned by server call to getTelescopeParams
    """
    now = Time.now().mjd
    hdr = create_header_from_telpars(telpars)

    # make dictionary of vals to put in table
    vals = {k: v for k, v in hdr.items() if k in VARIABLE_GTC_KEYS}
    vals['MJD'] = now
    # store LST as hourangle
    vals['LST'] = Longitude(vals['LST'], unit=u.hour).hourangle
    t.add_row(vals)