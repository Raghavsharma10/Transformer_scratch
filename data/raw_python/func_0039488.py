def recvall(self, timeout=0.5):
        """
        Receive the RCON command response
        :param timeout: The timeout between consequent data receive
        :return str: The RCON command response with header stripped out
        """
        response = ''
        self.socket.setblocking(False)
        start = time.time()
        while True:
            if response and time.time() - start > timeout:
                break
            elif time.time() - start > timeout * 2:
                break

            try:
                data = self.socket.recv(4096)
                if data:
                    response += data.replace(self._rconreplystring, '')
                    start = time.time()
                else:
                    time.sleep(0.1)
            except socket.error:
                pass

        return response.strip()