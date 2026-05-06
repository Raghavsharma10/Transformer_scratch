def update(self, configuration, debug=None):
        """Update the internal configuration values, removing debug_only
        handlers if debug is False. Returns True if the configuration has
        changed from previous configuration values.

        :param dict configuration: The logging configuration
        :param bool debug: Toggles use of debug_only loggers
        :rtype: bool

        """
        if self.config != dict(configuration) and debug != self.debug:
            self.config = dict(configuration)
            self.debug = debug
            self.configure()
            return True
        return False