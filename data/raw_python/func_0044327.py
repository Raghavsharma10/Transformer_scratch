def search(self, filepath=None, basedir=None, kind=None):
        """
        Search for a settings file.

        Keyword Arguments:
            filepath (string): Path to a config file, either absolute or
                relative. If absolute set its directory as basedir (omitting
                given basedir argument). If relative join it to basedir.
            basedir (string): Directory path where to search for.
            kind (string): Backend engine kind name (value of attribute
                ``_kind_name``) to help discovering with empty or relative
                filepath. Also if explicit absolute filepath is given, this
                will enforce the backend engine (such as yaml kind will be
                forced for a ``foo.json`` file).

        Returns:
            tuple: Absolute filepath and backend engine class.
        """
        # None values would cause trouble with path joining
        if filepath is None:
            filepath = ''
        if basedir is None:
            basedir = '.'

        if not basedir and not filepath:
            msg = "Either basedir or filepath is required for discovering"
            raise SettingsDiscoveryError(msg)

        if kind and kind not in self.engines:
            msg = "Given settings format is unknow: {}"
            raise SettingsDiscoveryError(msg.format(kind))

        # Implicit filename to find from backend
        if not filepath:
            filename, engine = self.guess_filename(basedir, kind)
            filepath = os.path.join(basedir, filename)
        # Explicit filename dont have to search for default backend file and
        # blindly force given backend if any
        else:
            if os.path.isabs(filepath):
                basedir, filename = os.path.split(filepath)
            else:
                filepath = os.path.join(basedir, filepath)

            if not os.path.exists(filepath):
                msg = "Given settings file does not exists: {}"
                raise SettingsDiscoveryError(msg.format(filepath))

            engine = self.get_engine(filepath, kind)

        return filepath, engine