def _loadRecordingDevices(self):
        """
        Loads the recordingDevices specified in the configuration.
        :param: handlers the loaded handlers.
        :return: the constructed recordingDevices in a dict keyed by name.
        """
        return {device.name: device for device in
                [self.createDevice(deviceCfg) for deviceCfg in self.config['accelerometers']]}