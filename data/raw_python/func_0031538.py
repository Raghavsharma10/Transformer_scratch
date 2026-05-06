def start_transmit(self, blocking=False, start_packet_groups=True, *ports):
        """ Start transmit on ports.

        :param blocking: True - wait for traffic end, False - return after traffic start.
        :param start_packet_groups: True - clear time stamps and start collecting packet groups stats, False - don't.
        :param ports: list of ports to start traffic on, if empty start on all ports.
        """

        port_list = self.set_ports_list(*ports)
        if start_packet_groups:
            port_list_for_packet_groups = self.ports.values()
            port_list_for_packet_groups = self.set_ports_list(*port_list_for_packet_groups)
            self.api.call_rc('ixClearTimeStamp {}'.format(port_list_for_packet_groups))
            self.api.call_rc('ixStartPacketGroups {}'.format(port_list_for_packet_groups))
        self.api.call_rc('ixStartTransmit {}'.format(port_list))
        time.sleep(0.2)

        if blocking:
            self.wait_transmit(*ports)