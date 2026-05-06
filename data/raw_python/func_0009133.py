def log(self, message, level=None):
        """Log a message to Storm optionally providing a logging level.

        :param message: the log message to send to Storm.
        :type message: str
        :param level: the logging level that Storm should use when writing the
                      ``message``. Can be one of: trace, debug, info, warn, or
                      error (default: ``info``).
        :type level: str

        .. warning::

          This will send your message to Storm regardless of what level you
          specify.  In almost all cases, you are better of using
          ``Component.logger`` and not setting ``pystorm.log.path``, because
          that will use a :class:`pystorm.component.StormHandler` to do the
          filtering on the Python side (instead of on the Java side after taking
          the time to serialize your message and send it to Storm).
        """
        level = _STORM_LOG_LEVELS.get(level, _STORM_LOG_INFO)
        self.send_message({"command": "log", "msg": str(message), "level": level})