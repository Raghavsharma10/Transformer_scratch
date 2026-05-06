def close(self):
        """Close any open channels."""
        self.consumer.close()
        self.publisher.close()
        self._closed = True