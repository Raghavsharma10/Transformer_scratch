def toFile(self, filename):
        """
        Save the suffix array instance including all features attached in
        filename. Accept any filename following the _open conventions,
        for example if it ends with .gz the file created will be a compressed
        GZip file.
        """
        start = _time()
        fd = _open(filename, "w")

        savedData = [self.string, self.unit, self.voc, self.vocSize, self.SA, self.features]

        for featureName in self.features:
            featureValues = getattr(self, "_%s_values" % featureName)
            featureDefault = getattr(self, "%s_default" % featureName)

            savedData.append((featureValues, featureDefault))

        fd.write(_dumps(savedData, _HIGHEST_PROTOCOL))
        fd.flush()
        try:
            self.sizeOfSavedFile = getsize(fd.name)
        except OSError:  # if stdout is used
            self.sizeOfSavedFile = "-1"
        self.toFileTime = _time() - start
        if _trace: print >> _stderr, "toFileTime %.2fs" % self.toFileTime

        if _trace: print >> _stderr, "sizeOfSavedFile %sb" % self.sizeOfSavedFile
        fd.close()