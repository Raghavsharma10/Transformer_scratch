def consume(self):
        """Declare consumers."""
        head = self.consumers[:-1]
        tail = self.consumers[-1]
        [self._declare_consumer(consumer, nowait=True)
                for consumer in head]
        self._declare_consumer(tail, nowait=False)