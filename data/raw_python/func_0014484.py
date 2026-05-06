def resize_pty(self, width=80, height=24):
        """
        Resize the pseudo-terminal.  This can be used to change the width and
        height of the terminal emulation created in a previous L{get_pty} call.

        @param width: new width (in characters) of the terminal screen
        @type width: int
        @param height: new height (in characters) of the terminal screen
        @type height: int

        @raise SSHException: if the request was rejected or the channel was
            closed
        """
        if self.closed or self.eof_received or self.eof_sent or not self.active:
            raise SSHException('Channel is not open')
        m = Message()
        m.add_byte(chr(MSG_CHANNEL_REQUEST))
        m.add_int(self.remote_chanid)
        m.add_string('window-change')
        m.add_boolean(True)
        m.add_int(width)
        m.add_int(height)
        m.add_int(0).add_int(0)
        self._event_pending()
        self.transport._send_user_message(m)
        self._wait_for_event()