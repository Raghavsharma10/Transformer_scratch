def send_jsonified(self, msg, stats=True):
        """ Send JSON-encoded message

        @param msg: JSON encoded string to send
        @param stats: If set to True, will update statistics after operation
                      completes
        """
        assert isinstance(msg, basestring), 'Can only send strings'

        if isinstance(msg, unicode):
            msg = msg.encode('utf-8')

        if self._immediate_flush:
            if self.handler and self.send_queue.is_empty():
                # Send message right away
                self.handler.send_pack('a[%s]' % msg)
            else:
                self.send_queue.push(msg)
                self.flush()
        else:
            self.send_queue.push(msg)

            if not self._pending_flush:
                reactor.callLater(0, self.flush)
                self._pending_flush = True

        if stats:
            self.stats.packSent(1)