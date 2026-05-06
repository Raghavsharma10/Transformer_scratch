def guess_filename(self, basedir, kind=None):
        """
        Try to find existing settings filename from base directory using
        default filename from available engines.

        First finded filename from available engines win. So registred engines
        order matter.

        Arguments:
            basedir (string): Directory path where to search for.

        Keyword Arguments:
            kind (string): Backend engine kind name to search for default
                settings filename. If not given, search will be made for
                default settings filename from all available backend engines.

        Returns:
            tuple: Absolute filepath and backend engine class.
        """
        if kind:
            filepath = os.path.join(basedir,
                                    self.engines[kind]._default_filename)
            if os.path.exists(filepath):
                return filepath, self.engines[kind]

        for filename, kind in self.filenames.items():
            filepath = os.path.join(basedir, filename)
            if os.path.exists(filepath):
                return filepath, self.engines[kind]

        msg = "Unable to find any settings in directory: {}"
        raise SettingsDiscoveryError(msg.format(basedir))