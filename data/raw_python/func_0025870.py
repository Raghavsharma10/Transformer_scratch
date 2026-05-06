def getMeasurements(self, measurementStatus=None):
        """
        Gets all available measurements.
        :param measurementStatus return only the measurements in the given state.
        :return:
        """
        if measurementStatus is None:
            return self.activeMeasurements + self.completeMeasurements + self.failedMeasurements
        elif measurementStatus == MeasurementStatus.COMPLETE:
            return self.completeMeasurements
        elif measurementStatus == MeasurementStatus.FAILED:
            return self.failedMeasurements
        elif measurementStatus == MeasurementStatus.DYING:
            return list(self.deathBed.keys())
        else:
            return [x for x in self.activeMeasurements if x.status == measurementStatus]