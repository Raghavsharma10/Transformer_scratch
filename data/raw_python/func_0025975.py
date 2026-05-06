def put(self, deviceId):
        """
        Puts a new device into the device store
        :param deviceId:
        :return:
        """
        device = request.get_json()
        logger.debug("Received /devices/" + deviceId + " - " + str(device))
        self._deviceController.accept(deviceId, device)
        return None, 200