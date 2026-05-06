def postJSON(g, data):
    """
    Posts the current setup to the camera and data servers.

    g : hcam_drivers.globals.Container
    Container with globals

    data : dict
    The current setup in JSON compatible dictionary format.
    """
    g.clog.debug('Entering postJSON')

    # encode data as json
    json_data = json.dumps(data).encode('utf-8')

    # Send the xml to the server
    url = urllib.parse.urljoin(g.cpars['hipercam_server'], g.SERVER_POST_PATH)
    g.clog.debug('Server URL = ' + url)

    opener = urllib.request.build_opener()
    g.clog.debug('content length = ' + str(len(json_data)))
    req = urllib.request.Request(url, data=json_data, headers={'Content-type': 'application/json'})
    response = opener.open(req, timeout=15).read()
    g.rlog.debug('Server response: ' + response.decode())
    csr = ReadServer(response, status_msg=False)
    if not csr.ok:
        g.clog.warn('Server response was not OK')
        g.rlog.warn('postJSON response: ' + response.decode())
        g.clog.warn('Server error = ' + csr.err)
        return False

    # now try to setup nodding server if appropriate
    if g.cpars['telins_name'] == 'GTC':
        url = urllib.parse.urljoin(g.cpars['gtc_offset_server'], 'setup')
        g.clog.debug('Offset Server URL = ' + url)
        opener = urllib.request.build_opener()
        try:
            req = urllib.request.Request(url, data=json_data, headers={'Content-type': 'application/json'})
            response = opener.open(req, timeout=5).read().decode()
        except Exception as err:
            g.clog.warn('Could not communicate with GTC offsetter')
            g.clog.warn(str(err))
            return False

        g.rlog.info('Offset Server Response: ' + response)
        if not json.loads(response)['status'] == 'OK':
            g.clog.warn('Offset Server response was not OK')
            return False

    g.clog.debug('Leaving postJSON')
    return True