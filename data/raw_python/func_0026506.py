def read(self, data):
        """Handles incoming raw sensor data and broadcasts it to specified
        udp servers and connected tcp clients
        :param data: NMEA raw sentences incoming data
        """

        self.log('Received NMEA data:', data, lvl=debug)
        # self.log(data, pretty=True)

        if self._tcp_socket is not None and \
                len(self._connected_tcp_endpoints) > 0:
            self.log('Publishing data on tcp server', lvl=debug)
            for endpoint in self._connected_tcp_endpoints:
                self.fireEvent(
                    write(
                        endpoint,
                        bytes(data, 'ascii')),
                    self.channel + '_tcp'
                )

        if self._udp_socket is not None and \
                len(self.config.udp_endpoints) > 0:
            self.log('Publishing data to udp endpoints', lvl=debug)
            for endpoint in self.config.udp_endpoints:
                host, port = endpoint.split(":")
                self.log('Transmitting to', endpoint, lvl=verbose)
                self.fireEvent(
                    write(
                        (host, int(port)),
                        bytes(data, 'ascii')
                    ),
                    self.channel +
                    '_udp'
                )