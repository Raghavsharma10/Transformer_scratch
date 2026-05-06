def close(self, code=None, reason=''):
        """
        Close the socket by sending a CLOSE frame and waiting for a response
        close message, unless such a message has already been received earlier
        (prior to calling this function, for example). The onclose() handler is
        called after the response has been received, but before the socket is
        actually closed.
        """
        self.send_close_frame(code, reason)

        frame = self.sock.recv()

        if frame.opcode != OPCODE_CLOSE:
            raise ValueError('expected CLOSE frame, got %s' % frame)

        self.handle_control_frame(frame)