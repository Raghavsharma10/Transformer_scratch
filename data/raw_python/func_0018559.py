def get_config(self, force=False):
        """
        Returns a dictionary of all config.xml properties

        If `force = True` then ignore any cached state and read config.xml
        if possible

        setup_omero_cli() must be called before this method to import the
        correct omero module to minimise the possibility of version conflicts
        """
        if not force and not self.has_config():
            raise Exception('No config file')

        configxml = os.path.join(self.dir, 'etc', 'grid', 'config.xml')
        if not os.path.exists(configxml):
            raise Exception('No config file')

        try:
            # Attempt to open config.xml read-only, though this flag is not
            # present in early versions of OMERO 5.0
            c = self._omero.config.ConfigXml(
                configxml, exclusive=False, read_only=True)
        except TypeError:
            c = self._omero.config.ConfigXml(configxml, exclusive=False)

        try:
            return c.as_map()
        finally:
            c.close()