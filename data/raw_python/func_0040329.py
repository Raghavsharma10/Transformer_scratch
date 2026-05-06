def close(self):
        """Close the generator."""
        if self.support_name:
            self.generator.close()
        try:
            next(self.generator)
        except StopIteration:
            return
        else:
            msg = "generator didn't stop: function {!r}"
            raise RuntimeError(msg.format(self.function))