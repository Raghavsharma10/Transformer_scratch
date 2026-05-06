def reserve_ports(self, ports_locations, force=False, clear=True, phy_mode=IxePhyMode.ignore):
        """ Reserve ports and reset factory defaults.

        :param ports_locations: list of ports ports_locations <ip, card, port> to reserve
        :param force: True - take forcefully, False - fail if port is reserved by other user
        :param clear: True - clear port configuration and statistics, False - leave port as is
        :param phy_mode: requested PHY mode.
        :return: ports dictionary (port uri, port object)
        """

        for port_location in ports_locations:
            ip, card, port = port_location.split('/')
            chassis = self.get_objects_with_attribute('chassis', 'ipAddress', ip)[0].id
            uri = '{} {} {}'.format(chassis, card, port)
            port = IxePort(parent=self, uri=uri)
            port._data['name'] = port_location
            port.reserve(force=force)
            if clear:
                port.clear()

        return self.ports