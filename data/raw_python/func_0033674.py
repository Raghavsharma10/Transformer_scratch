def run(self):
        """
        Process all incoming packets, until `stop()` is called. Intended to run
        in its own thread.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(self._addr)
        sock.settimeout(self._timeout)
        with closing(sock):
            while not self._shutdown.is_set():
                try:
                    data, addr = sock.recvfrom(self._buffer_size)
                except socket.timeout:
                    continue

                header, rest = parse_packet(data)
                if header.packet_type in _PAYLOADS:
                    payload = parse_payload(rest,
                                            *_PAYLOADS[header.packet_type])
                    self._callbacks.put(header.packet_type,
                                        header, payload, None, addr)
                else:
                    self._callbacks.put(EVENT_UNKNOWN,
                                        header, None, rest, addr)