def send_publish(self, mid, topic, payload, qos, retain, dup):
        """Send PUBLISH."""
        self.logger.debug("Send PUBLISH")
        if self.sock == NC.INVALID_SOCKET:
            return NC.ERR_NO_CONN

        #NOTE: payload may be any kind of data
        #      yet if it is a unicode string we utf8-encode it as convenience
        return self._do_send_publish(mid, utf8encode(topic), utf8encode(payload), qos, retain, dup)