def rm_docs(self):
        """Remove converted docs."""
        for filename in self.created:
            if os.path.exists(filename):
                os.unlink(filename)