def put_readme(self, content):
        """Store the readme descriptive metadata."""
        logger.debug("Putting readme")
        key = self.get_readme_key()
        self.put_text(key, content)