def _writeFeatures(self, i, image):
        """
        Write a text file containing the features as a table.

        @param i: The number of the image in self._images.
        @param image: A member of self._images.
        @return: The C{str} features file name - just the base name, not
            including the path to the file.
        """
        basename = 'features-%d.txt' % i
        filename = '%s/%s' % (self._outputDir, basename)
        featureList = image['graphInfo']['features']
        with open(filename, 'w') as fp:
            for feature in featureList:
                fp.write('%s\n\n' % feature.feature)
        return basename