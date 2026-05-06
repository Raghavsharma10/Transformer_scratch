def req(self, msgtype, payload, flags, size=0, offset=0, timeout=0):
        """send message to server and return response"""

        if timeout < 0:
            raise ValueError("timeout cannot be negative!")

        tohead = _ToServerHeader(payload=len(payload), type=msgtype,
                                 flags=flags, size=size, offset=offset)

        tstartcom = monotonic()  # set timer when communication begins
        self._send_msg(tohead, payload)

        while True:
            fromhead, data = self._read_msg()

            if fromhead.payload >= 0:
                # we received a valid answer and return the result
                return fromhead.ret, fromhead.flags, data

            assert msgtype != MSG_NOP

            # we did not exit the loop because payload is negative
            # Server said PING to keep connection alive during lenghty op

            # check if timeout has expired
            if timeout:
                tcom = monotonic() - tstartcom
                if tcom > timeout:
                    raise OwnetTimeout(tcom, timeout)