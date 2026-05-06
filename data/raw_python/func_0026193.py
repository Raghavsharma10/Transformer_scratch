def _handle_configspec(self, configspec):
        """Parse the configspec."""
        # FIXME: Should we check that the configspec was created with the
        #        correct settings ? (i.e. ``list_values=False``)
        if not isinstance(configspec, ConfigObj):
            try:
                configspec = ConfigObj(configspec,
                                       raise_errors=True,
                                       file_error=True,
                                       _inspec=True)
            except ConfigObjError as e:
                # FIXME: Should these errors have a reference
                #        to the already parsed ConfigObj ?
                raise ConfigspecError('Parsing configspec failed: %s' % e)
            except IOError as e:
                raise IOError('Reading configspec failed: %s' % e)

        self.configspec = configspec