def _refresh(self):
        """Background refreshing thread."""

        while not self._stopevent.isSet():
            line = self._serial.readline()
            #this is for python2/python3 compatibility. Is there a better way?
            try:
                line = line.encode().decode('utf-8')
            except AttributeError:
                line = line.decode('utf-8')

            if LaCrosseSensor.re_reading.match(line):
                sensor = LaCrosseSensor(line)
                self.sensors[sensor.sensorid] = sensor

                if self._callback:
                    self._callback(sensor, self._callback_data)

                if sensor.sensorid in self._registry:
                    for cbs in self._registry[sensor.sensorid]:
                        cbs[0](sensor, cbs[1])