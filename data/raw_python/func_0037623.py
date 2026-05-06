def reserve(self, location=None, force=False, wait_for_up=True, timeout=80):
        """ Reserve port and optionally wait for port to come up.

        :param location: port location as 'ip/module/port'. If None, the location will be taken from the configuration.
        :param force: whether to revoke existing reservation (True) or not (False).
        :param wait_for_up: True - wait for port to come up, False - return immediately.
        :param timeout: how long (seconds) to wait for port to come up.
        """

        if not location or is_local_host(location):
            return

        hostname, card, port = location.split('/')
        chassis = self.root.hw.get_chassis(hostname)

        # todo - test if port owned by me.
        if force:
            chassis.get_card(int(card)).get_port(int(port)).release()

        try:
            phy_port = chassis.get_card(int(card)).get_port(int(port))
        except KeyError as _:
            raise TgnError('Physical port {} unreachable'.format(location))
        self.set_attributes(commit=True, connectedTo=phy_port.ref)

        while self.get_attribute('connectedTo') == '::ixNet::OBJ-null':
            time.sleep(1)

        if wait_for_up:
            self.wait_for_up(timeout)