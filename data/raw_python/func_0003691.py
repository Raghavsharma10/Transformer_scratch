def configure(self, options, conf):
        """Configure which kinds of exceptions trigger plugin.
        """
        self.conf = conf
        self.enabled = options.epdb_debugErrors or options.epdb_debugFailures
        self.enabled_for_errors = options.epdb_debugErrors
        self.enabled_for_failures = options.epdb_debugFailures