def installedApp(self):
        """identify the propery application to launch, given the configuration"""
        try:    return self._installedApp
        except: # raises if not yet defined
            self._installedApp = runConfigs.get() # application/install/platform management
            return self._installedApp