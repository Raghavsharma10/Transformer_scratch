def start_capture(self, *ports):
        """ Start capture on ports.

        :param ports: list of ports to start capture on, if empty start on all ports.
        """

        IxeCapture.current_object = None
        IxeCaptureBuffer.current_object = None
        if not ports:
            ports = self.ports.values()
        for port in ports:
            port.captureBuffer = None
        port_list = self.set_ports_list(*ports)
        self.api.call_rc('ixStartCapture {}'.format(port_list))