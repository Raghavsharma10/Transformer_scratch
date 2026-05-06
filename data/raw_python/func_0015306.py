def register_console_logging_handler(cls, lgr, level=logging.INFO):
        """Registers console logging handler to given logger."""
        console_handler = logger.DevassistantClHandler(sys.stdout)
        if console_handler.stream.isatty():
            console_handler.setFormatter(logger.DevassistantClColorFormatter())
        else:
            console_handler.setFormatter(logger.DevassistantClFormatter())
        console_handler.setLevel(level)
        cls.cur_handler = console_handler
        lgr.addHandler(console_handler)