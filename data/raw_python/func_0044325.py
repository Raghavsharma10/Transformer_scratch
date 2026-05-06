def get_engine(self, filepath, kind=None):
        """
        From given filepath try to discover which backend format to use.

        Discovering is pretty naive as it find format from file extension.

        Args:
            filepath (str): Settings filepath or filename.

        Keyword Arguments:
            kind (str): A format name to enforce a specific backend. Can be any
                value from attribute ``_kind_name`` of available backend
                engines.

        Raises:
            boussole.exceptions.SettingsDiscoveryError: If extension is
            unknowed or if given format name is unknowed.

        Returns:
            object: Backend engine class.

        """
        if not kind:
            extension = os.path.splitext(filepath)[1]
            if not extension:
                msg = ("Unable to discover settings format from an empty file "
                       "extension: {}")
                raise SettingsDiscoveryError(msg.format(filepath))
            elif extension[1:] not in self.extensions:
                msg = ("Settings file extension is unknowed from available "
                       "backends: {}")
                raise SettingsDiscoveryError(msg.format(filepath))
            kind = self.extensions[extension[1:]]
        elif kind not in self.engines:
            msg = "Given settings format is unknow: {}"
            raise SettingsDiscoveryError(msg.format(kind))

        return self.engines[kind]