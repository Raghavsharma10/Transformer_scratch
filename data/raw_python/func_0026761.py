def tdSensorValue(self, protocol, model, sid, datatype):
        """Get the sensor value for a given sensor.

        :return: a dict with the keys: value, timestamp.
        """
        value = create_string_buffer(20)
        timestamp = c_int()

        self._lib.tdSensorValue(protocol, model, sid, datatype,
                                value, sizeof(value), byref(timestamp))
        return {'value': self._to_str(value), 'timestamp': timestamp.value}