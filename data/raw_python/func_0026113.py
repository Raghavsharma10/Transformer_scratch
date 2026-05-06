def _convertTmp(self, tmpCacheEntry):
        """
        Moves a tmp file to the upload dir, resampling it if necessary, and then deleting the tmp entries.
        :param tmpCacheEntry: the cache entry.
        :return:
        """
        from analyser.common.signal import loadSignalFromWav
        tmpCacheEntry['status'] = 'converting'
        logger.info("Loading " + tmpCacheEntry['path'])
        signal = loadSignalFromWav(tmpCacheEntry['path'])
        logger.info("Loaded " + tmpCacheEntry['path'])
        if Path(tmpCacheEntry['path']).exists():
            logger.info('Deleting ' + tmpCacheEntry['path'])
            os.remove(tmpCacheEntry['path'])
        else:
            logger.warning('Tmp cache file does not exist: ' + tmpCacheEntry['path'])
        self._tmpCache.remove(tmpCacheEntry)
        self._conversionCache.append(tmpCacheEntry)
        srcFs = signal.fs
        completeSamples = signal.samples
        outputFileName = os.path.join(self._uploadDir, tmpCacheEntry['name'])
        if srcFs > 1024:
            self.writeOutput(outputFileName, completeSamples, srcFs, 1000)
        else:
            self.writeOutput(outputFileName, completeSamples, srcFs, srcFs)
        tmpCacheEntry['status'] = 'loaded'
        self._conversionCache.remove(tmpCacheEntry)
        self._uploadCache.append(self._extractMeta(outputFileName, 'loaded'))