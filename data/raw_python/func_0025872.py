def store(self, measurement):
        """
        Writes the measurement metadata to disk on completion.
        :param activeMeasurement: the measurement that has completed.
        :returns the persisted metadata.
        """
        os.makedirs(self._getPathToMeasurementMetaDir(measurement.idAsPath), exist_ok=True)
        output = marshal(measurement, measurementFields)
        with open(self._getPathToMeasurementMetaFile(measurement.idAsPath), 'w') as outfile:
            json.dump(output, outfile)
        return output