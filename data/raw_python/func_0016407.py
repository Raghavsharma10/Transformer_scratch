def sendmess(self, msgtype, payload, flags=0, size=0, offset=0, timeout=0):
        """ retcode, data = sendmess(msgtype, payload)
        send generic message and returns retcode, data
        """

        flags |= self.flags
        assert not (flags & FLG_PERSISTENCE)

        with self._new_connection() as conn:
            ret, _, data = conn.req(
                msgtype, payload, flags, size, offset, timeout)

        return ret, data