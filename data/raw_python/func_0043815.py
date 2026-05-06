def launchApp(self, **kwargs):
        """Launch Starcraft2 process in the background using this configuration.
        WARNING: if the same IP address and port are specified between multiple
                 SC2 process instances, all subsequent processes after the first
                 will fail to initialize and crash.
        """
        app = self.installedApp
        # TODO -- launch host in window minimized/headless mode
        vers = self.getVersion()
        return app.start(version=vers,#game_version=vers.baseVersion, data_version=vers.dataHash,
            full_screen=self.fullscreen, verbose=self.debug, **kwargs)