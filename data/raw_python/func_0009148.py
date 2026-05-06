def process_batches(self):
        """Iterate through all batches, call process_batch on them, and ack.

        Separated out for the rare instances when we want to subclass
        BatchingBolt and customize what mechanism causes batches to be
        processed.
        """
        for key, batch in iteritems(self._batches):
            self._current_tups = batch
            self._current_key = key
            self.process_batch(key, batch)
            if self.auto_ack:
                for tup in batch:
                    self.ack(tup)
            # Set current batch to [] so that we know it was acked if a
            # later batch raises an exception
            self._current_key = None
            self._batches[key] = []
        self._batches = defaultdict(list)