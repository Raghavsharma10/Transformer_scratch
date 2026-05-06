def execute(self, duration):
        """
        Executes the measurement, recording the event status.
        :param duration: the time to run for.
        :return: nothing.
        """
        self.statuses.append({'name': ScheduledMeasurementStatus.RUNNING.name, 'time': datetime.utcnow()})
        try:
            self.recording = True
            self.device.start(self.name, durationInSeconds=duration)
        finally:
            self.recording = False
        if self.device.status == RecordingDeviceStatus.FAILED:
            self.statuses.append({'name': ScheduledMeasurementStatus.FAILED.name,
                                  'time': datetime.utcnow(),
                                  'reason': self.device.failureCode})
        else:
            self.statuses.append({'name': ScheduledMeasurementStatus.COMPLETE.name, 'time': datetime.utcnow()})
        # this is a bit of a hack, need to remove this at some point by refactoring the way measurements are stored
        if self.callback is not None:
            self.callback()