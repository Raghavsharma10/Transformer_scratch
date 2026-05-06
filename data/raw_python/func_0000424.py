def sensor(self, sensor_type):
        """Update and return sensor value."""
        _LOGGER.debug("Reading %s sensor.", sensor_type)
        return self._session.read_sensor(self.device_id, sensor_type)