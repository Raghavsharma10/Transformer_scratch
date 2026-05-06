def set_expanding(self):
        """Set that we are expanding a sequence, and return whether a release is required by the caller."""

        status = not self.expanding
        if status:
            self.expanding = True
        return status