def tcp_receive(self):
        """Receive data from TCP port."""
        data = self.conn.recv(self.BUFFER_SIZE)

        if type(data) != str:
            # Python 3 specific
            data = data.decode("utf-8")

        return str(data)