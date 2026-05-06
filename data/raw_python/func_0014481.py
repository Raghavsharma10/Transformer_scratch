def get_bytes(self, n):
        """
        Return the next C{n} bytes of the Message, without decomposing into
        an int, string, etc.  Just the raw bytes are returned.

        @return: a string of the next C{n} bytes of the Message, or a string
            of C{n} zero bytes, if there aren't C{n} bytes remaining.
        @rtype: string
        """
        b = self.packet.read(n)
        if len(b) < n:
            return b + '\x00' * (n - len(b))
        return b