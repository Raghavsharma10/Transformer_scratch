def stop_capture(self, cap_file_name=None, cap_file_format=IxeCapFileFormat.mem, *ports):
        """ Stop capture on ports.

        :param cap_file_name: prefix for the capture file name.
            Capture files for each port are saved as individual pcap file named 'prefix' + 'URI'.pcap.
        :param cap_file_format: exported file format
        :param ports: list of ports to stop traffic on, if empty stop all ports.
        :return: dictionary (port, nPackets)
        """

        port_list = self.set_ports_list(*ports)
        self.api.call_rc('ixStopCapture {}'.format(port_list))

        nPackets = {}
        for port in (ports if ports else self.ports.values()):
            nPackets[port] = port.capture.nPackets
            if nPackets[port]:
                if cap_file_format is not IxeCapFileFormat.mem:
                    port.cap_file_name = cap_file_name + '-' + port.uri.replace(' ', '_') + '.' + cap_file_format.name
                    port.captureBuffer.export(port.cap_file_name)
        return nPackets