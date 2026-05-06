def reserve_ports(self, locations, force=False, reset=True):
        """ Reserve ports and reset factory defaults.

        XenaManager-2G -> Reserve/Relinquish Port.
        XenaManager-2G -> Reset port.

        :param locations: list of ports locations in the form <module/port> to reserve
        :param force: True - take forcefully, False - fail if port is reserved by other user
        :param reset: True - reset port, False - leave port configuration
        :return: ports dictionary (index: object)
        """

        for location in locations:
            port = XenaPort(parent=self, index=location)
            port.reserve(force)
            if reset:
                port.reset()

        return self.ports