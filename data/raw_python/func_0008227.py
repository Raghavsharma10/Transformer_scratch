def createJSON(g, full=True):
    """
    Create JSON compatible dictionary from current settings

    Parameters
    ----------
    g :  hcam_drivers.globals.Container
    Container with globals
    """
    data = dict()
    if 'gps_attached' not in g.cpars:
        data['gps_attached'] = 1
    else:
        data['gps_attached'] = 1 if g.cpars['gps_attached'] else 0
    data['appdata'] = g.ipars.dumpJSON()
    data['user'] = g.rpars.dumpJSON()
    if full:
        data['hardware'] = g.ccd_hw.dumpJSON()
        data['tcs'] = g.info.dumpJSON()

        if g.cpars['telins_name'].lower() == 'gtc' and has_corba:
            try:
                s = get_telescope_server()
                data['gtc_headers'] = dict(
                    create_header_from_telpars(s.getTelescopeParams())
                )
            except:
                g.clog.warn('cannot get GTC headers from telescope server')
    return data