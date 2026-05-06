def run_forever(
        lcdproc='', mpd='', lcdproc_screen=DEFAULT_LCD_SCREEN_NAME,
        lcdproc_charset=DEFAULT_LCDPROC_CHARSET,
        lcdd_debug=False,
        pattern='', patterns=[],
        refresh=DEFAULT_REFRESH,
        backlight_on=DEFAULT_BACKLIGHT_ON,
        priority_playing=DEFAULT_PRIORITY,
        priority_not_playing=DEFAULT_PRIORITY,
        retry_attempts=DEFAULT_RETRY_ATTEMPTS,
        retry_wait=DEFAULT_RETRY_WAIT,
        retry_backoff=DEFAULT_RETRY_BACKOFF):
    """Run the server.

    Args:
        lcdproc (str): the target connection (host:port) for lcdproc
        mpd (str): the target connection ([pwd@]host:port) for mpd
        lcdproc_screen (str): the name of the screen to use for lcdproc
        lcdproc_charset (str): the charset to use with lcdproc
        lcdd_debug (bool): whether to enable full LCDd debug
        pattern (str): the pattern to use
        patterns (str list): the patterns to use
        refresh (float): how often to refresh the display
        backlight_on (str): the rules for activating backlight
        retry_attempts (int): number of connection attempts
        retry_wait (int): time between connection attempts
        retry_backoff (int): increase to between-attempts delay
    """
    # Compute host/ports
    lcd_conn = _make_hostport(lcdproc, 'localhost', 13666)
    mpd_conn = _make_hostport(mpd, 'localhost', 6600)

    # Prepare auto-retry
    retry_config = utils.AutoRetryConfig(
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        retry_wait=retry_wait)

    # Setup MPD client
    mpd_client = mpdwrapper.MPDClient(
        host=mpd_conn.hostname,
        port=mpd_conn.port,
        password=mpd_conn.username,
        retry_config=retry_config,
    )

    # Setup LCDd client
    lcd = _make_lcdproc(
        lcd_conn.hostname, lcd_conn.port,
        lcdd_debug=lcdd_debug,
        charset=lcdproc_charset,
        retry_config=retry_config,
    )

    # Setup connector
    runner = lcdrunner.MpdRunner(
        mpd_client, lcd,
        lcdproc_screen=lcdproc_screen,
        refresh_rate=refresh,
        retry_config=retry_config,
        backlight_on=backlight_on,
        priority_playing=priority_playing,
        priority_not_playing=priority_not_playing,
    )

    # Fill pattern
    if pattern:
        # If a specific pattern was given, use it
        patterns = [pattern]
    elif not patterns:
        # If no patterns were given, use the defaults
        patterns = DEFAULT_PATTERNS
    pattern_list = _make_patterns(patterns)

    mpd_hook_registry = mpdhooks.HookRegistry()
    runner.setup_pattern(pattern_list, hook_registry=mpd_hook_registry)

    # Launch
    mpd_client.connect()
    runner.run()

    # Exit
    logging.shutdown()