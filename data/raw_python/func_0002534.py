def send(self, packet_buffer):
        """
        send a buffer as a packet to the network interface
        :param packet_buffer: buffer to send (length shouldn't exceed MAX_INT)
        """
        if self._handle is None:
            raise self.DeviceIsNotOpen()
        buffer_length = len(packet_buffer)
        buf_send = ctypes.cast(ctypes.create_string_buffer(packet_buffer, buffer_length),
                               ctypes.POINTER(ctypes.c_ubyte))
        wtypes.pcap_sendpacket(self._handle, buf_send, buffer_length)