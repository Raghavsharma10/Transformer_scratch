def writeline(self, text):
        """Send a packet with line ending."""
        log.debug('writing line %r' % text)
        self.write(text+chr(10))