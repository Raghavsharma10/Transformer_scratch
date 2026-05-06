def process_tick(self, tick_tup):
        """Increment tick counter, and call ``process_batch`` for all current
        batches if tick counter exceeds ``ticks_between_batches``.

        See :class:`pystorm.component.Bolt` for more information.

        .. warning::
            This method should **not** be overriden.  If you want to tweak
            how Tuples are grouped into batches, override ``group_key``.
        """
        self._tick_counter += 1
        # ACK tick Tuple immediately, since it's just responsible for counter
        self.ack(tick_tup)
        if self._tick_counter > self.ticks_between_batches and self._batches:
            self.process_batches()
            self._tick_counter = 0