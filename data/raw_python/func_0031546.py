def set_prbs(self, rx_ports=None, tx_ports=None):
        """ Set TX ports and RX streams for stream statistics.

        :param ports: list of ports to set RX PRBS. If empty set for all ports.
        :type ports: list[ixexplorer.ixe_port.IxePort]
        :param tx_ports: list of streams to set TX PRBS. If empty set for all streams.
        :type tx_ports:  dict[ixexplorer.ixe_port.IxePort, list[ixexplorer.ixe_stream.IxeStream]]
        """

        if not rx_ports:
            rx_ports = self.ports.values()

        if not tx_ports:
            tx_ports = {}
            for port in self.ports.values():
                tx_ports[port] = port.streams.values()

        for port in rx_ports:
            port.set_receive_modes(IxeReceiveMode.widePacketGroup, IxeReceiveMode.sequenceChecking,
                                   IxeReceiveMode.prbs)
            port.enableAutoDetectInstrumentation = True
            port.autoDetectInstrumentation.ix_set_default()
            port.write()

        for port, streams in tx_ports.items():
            for stream in streams:
                stream.autoDetectInstrumentation.enableTxAutomaticInstrumentation = True
                stream.autoDetectInstrumentation.enablePRBS = True
            port.write()