def add_consumer(self, consumer):
        """Add another consumer from a :class:`Consumer` instance."""
        consumer.backend = self.backend
        self.consumers.append(consumer)