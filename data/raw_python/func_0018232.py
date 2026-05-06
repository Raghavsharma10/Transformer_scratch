def write(self, text):
        """Send a packet to the socket. This function cooks output."""
        text = str(text)    # eliminate any unicode or other snigglets
        text = text.replace(IAC, IAC+IAC)
        text = text.replace(chr(10), chr(13)+chr(10))
        self.writecooked(text)