def _on_auth(self, sock, authenticated):  # pylint: disable=unused-argument
        """Message received from websocket"""
        def ack(eventname, error, data):  # pylint: disable=unused-argument
            """Ack"""
            if error:
                self.log.error(f"""OnAuth: {error}""")
            else:
                self.connect_channels(self.channels)
                self.post_conn_cb()

        sock.emitack("auth", self.creds, ack)