def raw_data(self, event):
        """Handles incoming raw sensor data
        :param event: Raw sentences incoming data
        """

        self.log('Received raw data from bus', lvl=events)
        if not parse:
            return

        nmea_time = event.data[0]
        try:
            parsed_data = parse(event.data[1])
        except Exception as e:
            self.log('Unparseable sentence:', event.data[1], e, type(e),
                     exc=True, lvl=warn)
            self.unparsable += event
            return

        bus = event.bus

        sensor_data_package = self._handle(parsed_data)

        self.log("Sensor data:", sensor_data_package, lvl=verbose)

        if sensor_data_package:
            # pprint(sensor_data_package)
            self.fireEvent(sensordata(sensor_data_package, nmea_time, bus),
                           "navdata")