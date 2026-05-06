def run(self, callback=None, limit=0):
        """
        Start pcap's loop over the interface, calling the given callback for each packet
        :param callback: a function receiving (win_pcap, param, header, pkt_data) for each packet intercepted
        :param limit: how many packets to capture (A value of -1 or 0 is equivalent to infinity)
        """
        if self._handle is None:
            raise self.DeviceIsNotOpen()
        # Set new callback
        self._callback = callback
        # Run loop with callback wrapper
        wtypes.pcap_loop(self._handle, limit, self._callback_wrapper, None)