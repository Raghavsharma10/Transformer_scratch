def _make_lcdproc(
        lcd_host, lcd_port, retry_config,
        charset=DEFAULT_LCDPROC_CHARSET, lcdd_debug=False):
    """Create and connect to the LCDd server.

    Args:
        lcd_host (str): the hostname to connect to
        lcd_prot (int): the port to connect to
        charset (str): the charset to use when sending messages to lcdproc
        lcdd_debug (bool): whether to enable full LCDd debug
        retry_attempts (int): the number of connection attempts
        retry_wait (int): the time to wait between connection attempts
        retry_backoff (int): the backoff for increasing inter-attempt delay

    Returns:
        lcdproc.server.Server
    """

    class ServerSpawner(utils.AutoRetryCandidate):
        """Spawn the server, using auto-retry."""

        @utils.auto_retry
        def connect(self):
            return lcdrunner.LcdProcServer(
                lcd_host, lcd_port, charset=charset, debug=lcdd_debug)

    spawner = ServerSpawner(retry_config=retry_config, logger=logger)

    try:
        return spawner.connect()
    except socket.error as e:
        logger.error('Unable to connect to lcdproc %s:%s : %r', lcd_host, lcd_port, e)
        raise SystemExit(1)