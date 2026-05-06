def getVersion(self):
        """the executable application's version"""
        if isinstance(self.version, versions.Version):  return self.version
        if self.version: # verify specified version exists
            version = versions.Version(self.version) # create this object to allow self._version_ to be specified in multiple different ways by the user
            if version.baseVersion not in self.installedApp.versionMap(): # verify that the selected version has an executable
                raise runConfigs.lib.SC2LaunchError(
                    "specified game version %s executable is not available.%s    available:  %s"%( \
                    version, os.linesep, "  ".join(self.installedApp.listVersions())))
            self.version = version
        else: # get most recent executable's version
            path = self.installedApp.exec_path()
            vResult = self.installedApp.mostRecentVersion
            self.version = versions.Version(vResult)
        if self.debug: print(os.linesep.join([
            "Game configuration detail:",
            "    platform:   %s"%(self.os),
            "    app:        %s"%(self.execPath),
            "    version:    %s"%(self.version)]))
        return self.version