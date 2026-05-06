def _batch_entry_run(self):
        """The inside of ``_batch_entry``'s infinite loop.

        Separated out so it can be properly unit tested.
        """
        time.sleep(self.secs_between_batches)
        with self._batch_lock:
            self.process_batches()