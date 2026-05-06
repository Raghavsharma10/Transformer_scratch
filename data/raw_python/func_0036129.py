def _send_data(self, data):
        """Send data to the ADB server"""
        total_sent = 0

        while total_sent < len(data):
            # Send only the bytes that haven't been
            # sent yet
            sent = self.socket.send(data[total_sent:].encode("ascii"))

            if sent == 0:
                self.close()
                raise RuntimeError("Socket connection dropped, "
                                   "send failed")
            total_sent += sent