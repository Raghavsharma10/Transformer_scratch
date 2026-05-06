def update_ports(self):
        """
        Sets the `ports` attribute to the set of valid port values set in
        the configuration.
        """
        ports = set()

        for port in self.configured_ports:
            try:
                ports.add(int(port))
            except ValueError:
                logger.error("Invalid port value: %s", port)
                continue

        self.ports = ports