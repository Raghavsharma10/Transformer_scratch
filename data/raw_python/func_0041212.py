def unsubscribe(self, sid):
        """Disconnect an observer from this subject

        """
        if sid not in self.observers:
            raise KeyError(
                'Cannot disconnect a observer does not connected to subject'
            )
        del self.observers[sid]