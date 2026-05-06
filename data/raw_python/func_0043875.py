def start(self, version=None, **kwargs):#game_version=None, data_version=None, **kwargs):
    """Launch the game process."""
    if not version:
        version = self.mostRecentVersion
    pysc2Version = lib.Version( # convert to pysc2 Version
        version.version,
        version.baseVersion,
        version.dataHash,
        version.fixedHash)
    return sc_process.StarcraftProcess(
                self,
                exec_path=self.exec_path(version.baseVersion),
                version=pysc2Version,
                **kwargs)