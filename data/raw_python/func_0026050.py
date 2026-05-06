def patch(self, deviceId):
        """
        Updates the device with the given data. Supports a json payload like
        {
            fs: newFs
            samplesPerBatch: samplesPerBatch
            gyroEnabled: true
            gyroSensitivity: 500
            accelerometerEnabled: true
            accelerometerSensitivity: 2
        }
        A heartbeat is sent on completion of the request to ensure the analyser gets a rapid update.
        :return: the device and 200 if the update was ok, 400 if not.
        """
        try:
            device = self.recordingDevices.get(deviceId)
            if device.status == RecordingDeviceStatus.INITIALISED:
                errors = self._handlePatch(device)
                if len(errors) == 0:
                    return device, 200
                else:
                    return device, 500
            else:
                return device, 400
        finally:
            logger.info("Sending adhoc heartbeat on device state update")
            self.heartbeater.sendHeartbeat()