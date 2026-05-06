def sender(self):
        """
        Returns the sender, respecting the Resent-*
        headers. In any case, prefer Sender over From,
        meaning that if Sender is present then From is
        ignored, as per the RFC.
        """
        to_fetch = (
            ['Resent-Sender', 'Resent-From'] if self.resent else
            ['Sender', 'From']
        )
        for item in to_fetch:
            if item in self:
                _, addr = getaddresses([self[item]])[0]
                return addr