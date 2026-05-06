def reserve_ports(self, locations, force=False, reset=True):
        """ Reserve ports and reset factory defaults.

        XenaManager-2G -> Reserve/Relinquish Port.
        XenaManager-2G -> Reserve Port.

        :param locations: list of ports locations in the form <ip/slot/port> to reserve
        :param force: True - take forcefully. False - fail if port is reserved by other user
        :param reset: True - reset port, False - leave port configuration
        :return: ports dictionary (index: object)
        """

        for location in locations:
            ip, module, port = location.split('/')
            self.chassis_list[ip].reserve_ports(['{}/{}'.format(module, port)], force, reset)

        return self.ports