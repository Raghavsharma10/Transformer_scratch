def setup_omero_cli(self):
        """
        Imports the omero CLI module so that commands can be run directly.
        Note Python does not allow a module to be imported multiple times,
        so this will only work with a single omero instance.

        This can have several surprising effects, so setup_omero_cli()
        must be explcitly called.
        """
        if not self.dir:
            raise Exception('No server directory set')

        if 'omero.cli' in sys.modules:
            raise Exception('omero.cli can only be imported once')

        log.debug("Setting up omero CLI")
        lib = os.path.join(self.dir, "lib", "python")
        if not os.path.exists(lib):
            raise Exception("%s does not exist!" % lib)
        sys.path.insert(0, lib)

        import omero
        import omero.cli

        log.debug("Using omero CLI from %s", omero.cli.__file__)

        self.cli = omero.cli.CLI()
        self.cli.loadplugins()
        self._omero = omero