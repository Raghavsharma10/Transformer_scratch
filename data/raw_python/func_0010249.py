def get_data(self):
        """Get the contents of this file

        :return: The contents of this file
        :rtype: six.binary_type
        """

        target = DeviceTarget(self.device_id)
        return self._fssapi.get_file(target, self.path)[self.device_id]