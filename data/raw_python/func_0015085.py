def sensorupdate(self, data):
        """
        Given a dict of sensors and values, updates those sensors with the 
        values in Scratch.
        """
        if not isinstance(data, dict):
            raise TypeError('Expected a dict')
        msg = 'sensor-update '
        for key in data.keys():
            msg += '"%s" "%s" ' % (self._escape(str(key)), 
                                    self._escape(str(data[key])))
        self._send(msg)