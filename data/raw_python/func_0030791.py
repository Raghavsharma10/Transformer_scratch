def flush(self):
        """ Flush message queue if there's an active connection running """
        self._pending_flush = False

        if self.handler is None:
            return

        if self.send_queue.is_empty():
            return

        self.handler.send_pack('a[%s]' % self.send_queue.get())
        self.send_queue.clear()