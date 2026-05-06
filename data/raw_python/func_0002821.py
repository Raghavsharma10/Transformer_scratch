def config_logging(no_log_file, log_to, log_level, silent, verbosity):
    """
    Configures and generates a Logger object, 'openaccess_epub' based on common
    parameters used for console interface script execution in OpenAccess_EPUB.

    These parameters are:
      no_log_file
          Boolean. Disables logging to file. If set to True, log_to and
          log_level become irrelevant.
      log_to
          A string name indicating a file path for logging.
      log_level
          Logging level, one of: 'debug', 'info', 'warning', 'error', 'critical'
      silent
          Boolean
      verbosity
          Console logging level, one of: 'debug', 'info', 'warning', 'error',
          'critical

    This method currently only configures a console StreamHandler with a
    message-only Formatter.
    """

    log_level = get_level(log_level)
    console_level = get_level(verbosity)

    #We want to configure our openaccess_epub as the parent log
    log = logging.getLogger('openaccess_epub')
    log.setLevel(logging.DEBUG)  # Don't filter at the log level
    standard = logging.Formatter(STANDARD_FORMAT)
    message_only = logging.Formatter(MESSAGE_ONLY_FORMAT)

    #Only add FileHandler IF it's allowed AND we have a name for it
    if not no_log_file and log_to is not None:
        fh = logging.FileHandler(filename=log_to)
        fh.setLevel(log_level)
        fh.setFormatter(standard)
        log.addHandler(fh)

    #Add on the console StreamHandler at verbosity level if silent not set
    if not silent:
        sh_echo = logging.StreamHandler(sys.stdout)
        sh_echo.setLevel(console_level)
        sh_echo.setFormatter(message_only)
        log.addHandler(sh_echo)