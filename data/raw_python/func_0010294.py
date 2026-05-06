def get_devicecore_api(self):
        """Returns a :class:`.DeviceCoreAPI` bound to this device cloud instance

        This provides access to the same API as :attr:`.DeviceCloud.devicecore` but will create
        a new object (with a new cache) each time called.

        :return: devicecore API object bound to this device cloud account
        :rtype: :class:`.DeviceCoreAPI`

        """
        from devicecloud.devicecore import DeviceCoreAPI

        return DeviceCoreAPI(self._conn, self.get_sci_api())