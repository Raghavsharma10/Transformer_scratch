def wait_transmit(self, *ports):
        """ Wait for traffic end on ports.

        :param ports: list of ports to wait for, if empty wait for all ports.
        """

        port_list = self.set_ports_list(*ports)
        self.api.call_rc('ixCheckTransmitDone {}'.format(port_list))