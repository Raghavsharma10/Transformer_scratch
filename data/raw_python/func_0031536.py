def wait_for_up(self, timeout=16, ports=None):
        """ Wait until ports reach up state.

        :param timeout: seconds to wait.
        :param ports: list of ports to wait for.
        :return:
        """
        port_list = []
        for port in ports:
            port_list.append(self.set_ports_list(port))
        t_end = time.time() + timeout
        ports_not_in_up = []
        ports_in_up = []
        while time.time() < t_end:
            # ixCheckLinkState can take few seconds on some ports when link is down.
            for port in port_list:
                call = self.api.call('ixCheckLinkState {}'.format(port))
                if call == '0':
                    ports_in_up.append("{}".format(port))
                else:
                    pass
            ports_in_up = list(set(ports_in_up))
            if len(port_list) == len(ports_in_up):
                return
            time.sleep(1)
        for port in port_list:
            if port not in ports_in_up:
                ports_not_in_up.append(port)
        raise TgnError('{}'.format(ports_not_in_up))