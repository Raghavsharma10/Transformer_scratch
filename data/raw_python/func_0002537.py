def send_packet(self, pattern, packet_buffer, callback=None, limit=10):
        """
        Send a buffer as a packet to a network interface and optionally capture a response
        :param pattern: a wildcard pattern to match the description of a network interface to capture packets on
        :param packet_buffer: a buffer to send (length shouldn't exceed MAX_INT)
        :param callback: If not None, a function to call with each intercepted packet
        :param limit: how many packets to capture (A value of -1 or 0 is equivalent to infinity)
        """
        device_name, desc = WinPcapDevices.get_matching_device(pattern)
        if device_name is not None:
            with WinPcap(device_name) as capture:
                capture.send(packet_buffer)
                if callback is not None:
                    capture.run(callback=callback, limit=limit)