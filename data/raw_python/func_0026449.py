def serial_packet(self, event):
        """Handles incoming raw sensor data
        :param data: raw incoming data
        """

        self.log('Incoming serial packet:', event.__dict__, lvl=verbose)

        if self.scanning:
            pass
        else:
            # self.log("Incoming data: ", '%.50s ...' % event.data, lvl=debug)
            sanitized_data = self._parse(event.bus, event.data)
            self.log('Sanitized data:', sanitized_data, lvl=verbose)
            if sanitized_data is not None:
                self._broadcast(event.bus, sanitized_data)