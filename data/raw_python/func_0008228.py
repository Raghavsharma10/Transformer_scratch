def insertFITSHDU(g):
    """
    Uploads a table of TCS data to the servers, which is appended onto a run.

    Arguments
    ---------
    g : hcam_drivers.globals.Container
        the Container object of application globals
    """
    if not g.cpars['hcam_server_on']:
        g.clog.warn('insertFITSHDU: servers are not active')
        return False

    run_number = getRunNumber(g)
    tcs_table = g.info.tcs_table

    g.clog.info('Adding TCS table data to run{:04d}.fits'.format(run_number))
    url = g.cpars['hipercam_server'] + 'addhdu'
    try:
        fd = StringIO()
        ascii.write(tcs_table, format='ecsv', output=fd)
        files = {'file': fd.getvalue()}
        r = requests.post(url, data={'run': 'run{:04d}.fits'.format(run_number)},
                          files=files)
        fd.close()
        rs = ReadServer(r.content, status_msg=False)
        if rs.ok:
            g.clog.info('Response from server was OK')
            return True
        else:
            g.clog.warn('Response from server was not OK')
            g.clog.warn('Reason: ' + rs.err)
            return False
    except Exception as err:
        g.clog.warn('insertFITSHDU failed')
        g.clog.warn(str(err))