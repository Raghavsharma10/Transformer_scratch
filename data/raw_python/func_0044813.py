def save(self, reload=False):
        """Save changes to the file."""
        self.wrapper.raw.save()
        if reload:
            self.reload()