def set_options(self, multiplex_first=True, **bogus_options):
        "Takes implementation specific options. To be overriden in a subclass."
        self.multiplex_first = multiplex_first
        self._warn_bogus_options(**bogus_options)