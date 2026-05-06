def accept(self, deviceId, device):
        """
        Adds the named device to the store.
        :param deviceId:
        :param device:
        :return:
        """
        storedDevice = self.devices.get(deviceId)
        if storedDevice is None:
            logger.info('Initialising device ' + deviceId)
            storedDevice = Device(self.maxAgeSeconds)
            storedDevice.deviceId = deviceId
            # this uses an async handler to decouple the recorder put (of the data) from the analyser handling that data
            # thus the recorder will become free as soon as it has handed off the data. This means delivery is only
            # guaranteed as long as the analyser stays up but this is not a system that sits on top of a bulletproof
            # message bus so unlucky :P
            storedDevice.dataHandler = AsyncHandler('analyser', CSVLogger('analyser', deviceId, self.dataDir))
        else:
            logger.debug('Pinged by device ' + deviceId)
        storedDevice.payload = device
        storedDevice.lastUpdateTime = datetime.datetime.utcnow()
        # TODO if device has FAILED, do something?
        self.devices.update({deviceId: storedDevice})
        self.targetStateController.updateDeviceState(storedDevice.payload)