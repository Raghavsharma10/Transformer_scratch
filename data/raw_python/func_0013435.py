def _scanDataFiles(self, dataDir, patterns):
        """
        Scans the specified directory for files with the specified globbing
        pattern and calls self._addDataFile for each. Raises an
        EmptyDirException if no data files are found.
        """
        numDataFiles = 0
        for pattern in patterns:
            scanPath = os.path.join(dataDir, pattern)
            for filename in glob.glob(scanPath):
                self._addDataFile(filename)
                numDataFiles += 1
        if numDataFiles == 0:
            raise exceptions.EmptyDirException(dataDir, patterns)