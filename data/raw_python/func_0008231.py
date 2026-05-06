def getFrameNumber(g):
    """
    Polls the data server to find the current frame number.

    Throws an exceotion if it cannot determine it.
    """
    if not g.cpars['hcam_server_on']:
        raise DriverError('getRunNumber error: servers are not active')
    url = g.cpars['hipercam_server'] + 'status/DET.FRAM2.NO'
    response = urllib.request.urlopen(url, timeout=2)
    rs = ReadServer(response.read(), status_msg=False)
    try:
        msg = rs.msg
    except:
        raise DriverError('getFrameNumber error: no message found')
    try:
        frame_no = int(msg.split()[1])
    except:
        raise DriverError('getFrameNumber error: invalid msg ' + msg)
    return frame_no