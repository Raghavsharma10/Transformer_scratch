def execPath(self):
        """the executable application's path"""
        vers = self.version.label if self.version else None # executables in Versions folder are stored by baseVersion (modified by game data patches)
        return self.installedApp.exec_path(vers)