def exec_path(self, baseVersion=None):
    """Get the exec_path for this platform. Possibly find the latest build."""
    if not os.path.isdir(self.data_dir):
        raise sc_process.SC2LaunchError("Install Starcraft II at %s or set the SC2PATH environment variable"%(self.data_dir))
    if baseVersion==None: # then select most recent version's baseVersion
        mostRecent = versions.handle.mostRecent
        if mostRecent:  return mostRecent["base-version"]
        raise sc_process.SC2LaunchError(
            "When requesting a versioned executable path without specifying base-version, expected "
            "to find StarCraft II versions installed at %s."%(self.versionsDir))
    elif isinstance(baseVersion, versions.Version):
        baseVersion = baseVersion.baseVersion
    elif str(baseVersion).count(".") > 0:
        baseVersion = versions.Version(baseVersion).baseVersion
    #else: # otherwise expect that the baseVersion specified is correct
    baseVersExec = os.path.join(self.versionsDir, "Base%s"%baseVersion, self._exec_name)
    if os.path.isfile(baseVersExec):
        return baseVersExec # if baseVersion in Versions subdir is valid, it is the correct executable
    raise sc_process.SC2LaunchError("Specified baseVersion %s does not exist at %s.%s    available: %s"%(\
        baseVersion, baseVersExec, os.linesep, " ".join(
            str(val) for val in sorted(self.versionMap().keys())) ))