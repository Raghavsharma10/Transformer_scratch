def updateDeviceState(self, device):
        """
        Updates the target state on the specified device.
        :param targetState: the target state to reach.
        :param device: the device to update.
        :return:
        """
        # this is only threadsafe because the targetstate is effectively immutable, if it becomes mutable in future then
        # funkiness may result
        self._reactor.offer(REACH_TARGET_STATE, [self._targetStateProvider.state, device, self._httpclient])