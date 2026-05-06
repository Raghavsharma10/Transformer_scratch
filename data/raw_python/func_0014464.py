def send_ignore(self, bytes=None):
        """
        Send a junk packet across the encrypted link.  This is sometimes used
        to add "noise" to a connection to confuse would-be attackers.  It can
        also be used as a keep-alive for long lived connections traversing
        firewalls.

        @param bytes: the number of random bytes to send in the payload of the
            ignored packet -- defaults to a random number from 10 to 41.
        @type bytes: int
        """
        m = Message()
        m.add_byte(chr(MSG_IGNORE))
        if bytes is None:
            bytes = (ord(rng.read(1)) % 32) + 10
        m.add_bytes(rng.read(bytes))
        self._send_user_message(m)