def add_consumer_from_dict(self, queue, **options):
        """Add another consumer from dictionary configuration."""
        options.setdefault("routing_key", options.pop("binding_key", None))
        consumer = Consumer(self.connection, queue=queue,
                            backend=self.backend, **options)
        self.consumers.append(consumer)
        return consumer