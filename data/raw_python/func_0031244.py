def set_transmit_mode(self, mode):
        """ set port transmit mode

        :param mode: request transmit mode
        :type mode: ixexplorer.ixe_port.IxeTransmitMode
        """

        self.api.call_rc('port setTransmitMode {} {}'.format(mode, self.uri))