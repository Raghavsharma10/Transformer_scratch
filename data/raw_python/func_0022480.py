def conditions(self):
        """The if-else pairs."""
        for idx in six.moves.range(1, len(self.children), 2):
            yield (self.children[idx - 1], self.children[idx])