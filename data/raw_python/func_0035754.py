def put_admin_metadata(self, admin_metadata):
        """Store the admin metadata."""
        logger.debug("Putting admin metdata")
        text = json.dumps(admin_metadata)
        key = self.get_admin_metadata_key()
        self.put_text(key, text)