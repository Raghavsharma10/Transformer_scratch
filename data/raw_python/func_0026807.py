def value(self, datatype):
        """Return the :class:`SensorValue` for the given data type.

        sensor.value(TELLSTICK_TEMPERATURE) is identical to calling
        sensor.temperature().
        """
        value = self.lib.tdSensorValue(
            self.protocol, self.model, self.id, datatype)
        return SensorValue(datatype, value['value'], value['timestamp'])