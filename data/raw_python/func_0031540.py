def stop_transmit(self, *ports):
        """ Stop traffic on ports.

        :param ports: list of ports to stop traffic on, if empty start on all ports.
        """

        port_list = self.set_ports_list(*ports)
        self.api.call_rc('ixStopTransmit {}'.format(port_list))
        time.sleep(0.2)