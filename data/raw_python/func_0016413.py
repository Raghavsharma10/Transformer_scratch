def sendmess(self, msgtype, payload, flags=0, size=0, offset=0, timeout=0):
        """
        retcode, data = sendmess(msgtype, payload)
        send generic message and returns retcode, data
        """

        # reuse last valid connection or create new
        conn = self.conn or self._new_connection()
        # invalidate last connection
        self.conn = None

        flags |= self.flags
        assert (flags & FLG_PERSISTENCE)
        ret, rflags, data = conn.req(
            msgtype, payload, flags, size, offset, timeout)
        if rflags & FLG_PERSISTENCE:
            # persistence granted, save connection object for reuse
            self.conn = conn
        else:
            # discard connection object
            conn.shutdown()

        return ret, data