def set_stream_stats(self, rx_ports=None, tx_ports=None, start_offset=40,
                         sequence_checking=True, data_integrity=True, timestamp=True):
        """ Set TX ports and RX streams for stream statistics.

        :param ports: list of ports to set RX pgs. If empty set for all ports.
        :type ports: list[ixexplorer.ixe_port.IxePort]
        :param tx_ports: list of streams to set TX pgs. If empty set for all streams.
        :type tx_ports:  dict[ixexplorer.ixe_port.IxePort, list[ixexplorer.ixe_stream.IxeStream]]
        :param sequence_checking: True - enable sequence checkbox, False - disable
        :param data_integrity: True - enable data integrity checkbox, False - disable
        :param timestamp: True - enable timestamp checkbox, False - disable
        :param start_offset: start offset for signatures (group ID, signature, sequence)
        """

        if not rx_ports:
            rx_ports = self.ports.values()

        if not tx_ports:
            tx_ports = {}
            for port in self.ports.values():
                tx_ports[port] = port.streams.values()

        groupIdOffset = start_offset
        signatureOffset = start_offset + 4
        next_offset = start_offset + 8
        if sequence_checking:
            sequenceNumberOffset = next_offset
            next_offset += 4
        if data_integrity:
            di_signatureOffset = next_offset

        for port in rx_ports:
            modes = []
            modes.append(IxeReceiveMode.widePacketGroup)
            port.packetGroup.groupIdOffset = groupIdOffset
            port.packetGroup.signatureOffset = signatureOffset
            if sequence_checking and int(port.isValidFeature('portFeatureRxSequenceChecking')):
                modes.append(IxeReceiveMode.sequenceChecking)
                port.packetGroup.sequenceNumberOffset = sequenceNumberOffset
            if data_integrity and int(port.isValidFeature('portFeatureRxDataIntegrity')):
                modes.append(IxeReceiveMode.dataIntegrity)
                port.dataIntegrity.signatureOffset = di_signatureOffset
            if timestamp and int(port.isValidFeature('portFeatureRxFirstTimeStamp')):
                port.dataIntegrity.enableTimeStamp = True
            else:
                port.dataIntegrity.enableTimeStamp = False
            port.set_receive_modes(*modes)

            port.write()

        for port, streams in tx_ports.items():
            for stream in streams:
                stream.packetGroup.insertSignature = True
                stream.packetGroup.groupIdOffset = groupIdOffset
                stream.packetGroup.signatureOffset = signatureOffset
                if sequence_checking:
                    stream.packetGroup.insertSequenceSignature = True
                    stream.packetGroup.sequenceNumberOffset = sequenceNumberOffset
                if data_integrity and int(port.isValidFeature('portFeatureRxDataIntegrity')):
                    stream.dataIntegrity.insertSignature = True
                    stream.dataIntegrity.signatureOffset = di_signatureOffset
                if timestamp:
                    stream.enableTimestamp = True
                else:
                    stream.enableTimestamp = False

            port.write()