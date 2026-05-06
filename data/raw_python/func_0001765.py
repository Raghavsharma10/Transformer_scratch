def setup_logger(log_level, log_file=None, logger_name=None):
        """setup logger
            @param log_level: debug/info/warning/error/critical
            @param log_file: log file path
            @param logger_name: the name of logger, default is 'root' if not specify
        """
        applogger = AppLog(logger_name)
        level = getattr(logging, log_level.upper(), None)
        if not level:
            color_print("Invalid log level: %s" % log_level, "RED")
            sys.exit(1)
    
        # hide traceback when log level is INFO/WARNING/ERROR/CRITICAL
        if level >= logging.INFO:
            sys.tracebacklimit = 0
    
        if log_file:
            applogger._handle2file(log_file)
        else:
            applogger._handle2screen(color = True)
        
        applogger.logger.setLevel(level)