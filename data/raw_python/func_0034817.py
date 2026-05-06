def handle_control_frame(self, frame):
        """
        Handle a control frame as defined by RFC 6455.
        """
        if frame.opcode == OPCODE_CLOSE:
            self.close_frame_received = True
            code, reason = frame.unpack_close()

            if self.close_frame_sent:
                self.onclose(code, reason)
                self.sock.close()
                raise SocketClosed(True)
            else:
                self.close_params = (code, reason)
                self.send_close_frame(code, reason)

        elif frame.opcode == OPCODE_PING:
            # Respond with a pong message with identical payload
            self.send_frame(ControlFrame(OPCODE_PONG, frame.payload))

        elif frame.opcode == OPCODE_PONG:
            # Assert that the PONG payload is identical to that of the PING
            if not self.ping_sent:
                raise PingError('received PONG while no PING was sent')

            self.ping_sent = False

            if frame.payload != self.ping_payload:
                raise PingError('received PONG with invalid payload')

            self.ping_payload = None
            self.onpong(frame.payload)