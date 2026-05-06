def receivers(self):
        """
        Returns a list of receivers, obtained from the
        To, Cc, and Bcc headers, respecting the Resent-*
        headers if the email was resent.
        """
        attrs = (
            ['Resent-To', 'Resent-Cc', 'Resent-Bcc'] if self.resent else
            ['To', 'Cc', 'Bcc']
        )
        addrs = (v for v in (self.get(k) for k in attrs) if v)
        return [addr for _, addr in getaddresses(addrs)]