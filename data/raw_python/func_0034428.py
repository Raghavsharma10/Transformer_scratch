def new_log_level(level, name, logger_name=None):
    """
    Quick way to create a custom log level that behaves like the default levels in the logging module.
    :param level: level number
    :param name: level name
    :param logger_name: optional logger name
    """
    @CustomLogLevel(level, name, logger_name)
    def _default_template(logger, msg, *args, **kwargs):
        return msg, args, kwargs