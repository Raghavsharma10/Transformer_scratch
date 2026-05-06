def _clashes(self, startTime, duration):
        """
        verifies that this measurement does not clash with an already scheduled measurement.
        :param startTime: the start time.
        :param duration: the duration.
        :return: true if the measurement is allowed.
        """
        return [m for m in self.activeMeasurements if m.overlapsWith(startTime, duration)]