def start_packet_groups(self, clear_time_stamps=True, *ports):
        """ Start packet groups on ports.

        :param clear_time_stamps: True - clear time stamps, False - don't.
        :param ports: list of ports to start traffic on, if empty start on all ports.
        """
        port_list = self.set_ports_list(*ports)
        if clear_time_stamps:
            self.api.call_rc('ixClearTimeStamp {}'.format(port_list))
        self.api.call_rc('ixStartPacketGroups {}'.format(port_list))