def ping(self):
        """sends a NOP packet and waits response; returns None"""

        ret, data = self.sendmess(MSG_NOP, bytes())
        if data or ret > 0:
            raise ProtocolError('invalid reply to ping message')
        if ret < 0:
            raise OwnetError(-ret, self.errmess[-ret])