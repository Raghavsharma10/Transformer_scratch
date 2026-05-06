def execCommand(g, command, timeout=10):
    """
    Executes a command by sending it to the rack server

    Arguments:
      g : hcam_drivers.globals.Container
        the Container object of application globals
      command : (string)
           the command (see below)

    Possible commands are:

      start   : starts a run
      stop    : stops a run
      abort   : aborts a run
      online  : bring ESO control server online and power up hardware
      off     : put ESO control server in idle state and power down
      standby : server can communicate, but child processes disabled
      reset   : resets the NGC controller front end

    Returns True/False according to whether the command
    succeeded or not.
    """
    if not g.cpars['hcam_server_on']:
        g.clog.warn('execCommand: servers are not active')
        return False

    try:
        url = g.cpars['hipercam_server'] + command
        g.clog.info('execCommand, command = "' + command + '"')
        response = urllib.request.urlopen(url, timeout=timeout)
        rs = ReadServer(response.read(), status_msg=False)

        g.rlog.info('Server response =\n' + rs.resp())
        if rs.ok:
            g.clog.info('Response from server was OK')
            return True
        else:
            g.clog.warn('Response from server was not OK')
            g.clog.warn('Reason: ' + rs.err)
            return False
    except urllib.error.URLError as err:
        g.clog.warn('execCommand failed')
        g.clog.warn(str(err))

    return False