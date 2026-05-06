def disconnect(self, receiver):
        """Remove receiver."""
        try:
            self.receivers.remove(receiver)
        except ValueError:
            raise ValueError('Unknown receiver: %s' % receiver)