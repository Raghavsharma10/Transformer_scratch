def inflate(self):
        """
        loads the recording into memory and returns it as a Signal
        :return:
        """
        if self.measurementParameters['accelerometerEnabled']:
            if len(self.data) == 0:
                logger.info('Loading measurement data for ' + self.name)
                self.data = {name: self._loadXYZ(name) for name, value in self.recordingDevices.items()}
            return True
        else:
            # TODO error handling
            return False