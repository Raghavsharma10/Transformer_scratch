def unpack_close(self):
        """
        Unpack a close message into a status code and a reason. If no payload
        is given, the code is None and the reason is an empty string.
        """
        if self.payload:
            code = struct.unpack('!H', str(self.payload[:2]))[0]
            reason = str(self.payload[2:])
        else:
            code = None
            reason = ''

        return code, reason