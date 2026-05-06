def updateDeviceStatus(self, deviceName, state, reason=None):
        """
        Updates the current device status.
        :param deviceName: the device name.
        :param state: the state.
        :param reason: the reason for the change.
        :return:
        """
        logger.info('Updating recording device state for ' + deviceName + ' to ' + state.name +
                    ('' if reason is None else '[reason: ' + reason + ']'))
        currentState = self.recordingDevices.get(deviceName)
        count = 0
        if currentState is not None:
            if currentState['state'] == MeasurementStatus.RECORDING.name:
                count = currentState['count']
        self.recordingDevices[deviceName] = {
            'state': state.name,
            'reason': reason,
            'time': datetime.datetime.utcnow().strftime(DATETIME_FORMAT),
            'count': count
        }