def connect(self, receiver):
        """Append receiver."""
        if not callable(receiver):
            raise ValueError('Invalid receiver: %s' % receiver)
        self.receivers.append(receiver)