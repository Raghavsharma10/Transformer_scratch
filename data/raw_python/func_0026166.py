def updateTargetState(self, newState):
        """
        Updates the system target state and propagates that to all devices.
        :param newState:
        :return:
        """
        self._targetStateProvider.state = loadTargetState(newState, self._targetStateProvider.state)
        for device in self.deviceController.getDevices():
            self.updateDeviceState(device.payload)