def _activate(self):
        """Activates the stream."""
        if six.callable(self.streamer):
            # If it's a function, create the stream.
            self.stream_ = self.streamer(*(self.args), **(self.kwargs))

        else:
            # If it's iterable, use it directly.
            self.stream_ = iter(self.streamer)