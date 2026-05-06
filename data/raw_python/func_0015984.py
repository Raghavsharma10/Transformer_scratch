def _disable(self):
        """
        The configuration containing this function has been disabled by host.
        Endpoint do not work anymore, so cancel AIO operation blocks.
        """
        if self._enabled:
            self._real_onCannotSend()
            has_cancelled = 0
            for block in self._aio_recv_block_list + self._aio_send_block_list:
                try:
                    self._aio_context.cancel(block)
                except OSError as exc:
                    trace(
                        'cancelling %r raised: %s' % (block, exc),
                    )
                else:
                    has_cancelled += 1
            if has_cancelled:
                noIntr(functools.partial(self._aio_context.getEvents, min_nr=None))
            self._enabled = False