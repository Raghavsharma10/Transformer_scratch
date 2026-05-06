def overlapsWith(self, targetStartTime, duration):
        """
        Tests if the given times overlap with this measurement.
        :param targetStartTime: the target start time.
        :param duration: the duration
        :return: true if the given times overlap with this measurement.
        """
        targetEndTime = targetStartTime + datetime.timedelta(days=0, seconds=duration)
        return (self.startTime <= targetStartTime <= self.endTime) \
               or (targetStartTime <= self.startTime <= targetEndTime)