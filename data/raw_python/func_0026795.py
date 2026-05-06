def sensors(self):
        """Return all known sensors.

        :return: list of :class:`Sensor` instances.
        """
        sensors = []
        try:
            while True:
                sensor = self.lib.tdSensor()
                sensors.append(Sensor(lib=self.lib, **sensor))
        except TelldusError as e:
            if e.error != const.TELLSTICK_ERROR_DEVICE_NOT_FOUND:
                raise
        return sensors