def load(self, path):
        """
        Loads a CompletedMeasurement from the path.á
        :param path: the path at which the data is found.
        :return: the measurement
        """
        meta = self._loadMetaFromJson(path)
        return CompleteMeasurement(meta, self.dataDir) if meta is not None else None