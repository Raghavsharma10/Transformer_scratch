def run(self):
        """
        Process all outgoing packets, until `stop()` is called. Intended to run
        in its own thread.
        """
        while True:
            to_send = self._queue.get()
            if to_send is _SHUTDOWN:
                break

            # If we get a gateway object, connect to it. Otherwise, assume
            # it's a bytestring and send it out on the socket.
            if isinstance(to_send, Gateway):
                self._gateway = to_send
                self._connected.set()
            else:
                if not self._gateway:
                    raise SendException('no gateway')
                dest = (self._gateway.addr, self._gateway.port)
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(to_send, dest)