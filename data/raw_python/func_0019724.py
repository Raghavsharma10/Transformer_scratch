def getFilesystemStats(self, fs):
        """Returns I/O stats for filesystem.
        
        @param fs: Filesystem path.
        @return: Dict of stats.
        
        """
        if self._mapFSpathDev is None:
            self._initFilesystemInfo()
        return self._diskStats.get(self._mapFSpathDev.get(fs))