def reserve(self, ports, force=False, wait_for_up=True, timeout=80):
        """ Reserve port and optionally wait for port to come up.

        :param ports: dict of <port, ip/module/port'>.
        :param force: whether to revoke existing reservation (True) or not (False).
        :param wait_for_up: True - wait for port to come up, False - return immediately.
        :param timeout: how long (seconds) to wait for port to come up.
        """

        if force:
            for port in ports:
                port.release()

        for port, location in ports.items():
            port.reserve(location, False, wait_for_up, timeout)