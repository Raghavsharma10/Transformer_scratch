def send_ping(self, payload=''):
        """
        Send a PING control frame with an optional payload.
        """
        self.send_frame(ControlFrame(OPCODE_PING, payload),
                        lambda: self.onping(payload))
        self.ping_payload = payload
        self.ping_sent = True