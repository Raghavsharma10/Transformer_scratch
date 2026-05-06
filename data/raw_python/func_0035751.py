def get_admin_metadata(self):
        """Return the admin metadata as a dictionary."""
        logger.debug("Getting admin metdata")
        text = self.get_text(self.get_admin_metadata_key())
        return json.loads(text)