def _sendDDEcommand(self, cmd, timeout=None):
        """Send command to DDE client"""
        reply = self.conversation.Request(cmd, timeout)
        if self.pyver > 2:
            reply = reply.decode('ascii').rstrip()
        return reply