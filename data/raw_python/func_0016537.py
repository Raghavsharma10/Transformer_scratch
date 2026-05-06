def iterconsume(self, limit=None):
        """Cycle between all consumers in consume mode.

        See :meth:`Consumer.iterconsume`.
        """
        self.consume()
        return self.backend.consume(limit=limit)