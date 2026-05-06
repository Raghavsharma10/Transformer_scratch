def getDevices(self, status=None):
        """
        The devices in the given state or all devices is the arg is none.
        :param status: the state to match against.
        :return: the devices
        """
        return [d for d in self.devices.values() if status is None or d.payload.get('status') == status]