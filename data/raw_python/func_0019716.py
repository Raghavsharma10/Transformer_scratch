def _initFilesystemInfo(self):
        """Initialize filesystem to device mappings."""
        self._mapFSpathDev = {}
        fsinfo = FilesystemInfo()
        for fs in fsinfo.getFSlist():
            devpath = fsinfo.getFSdev(fs)
            dev = self._getUniqueDev(devpath)
            if dev is not None:
                self._mapFSpathDev[fs] = dev