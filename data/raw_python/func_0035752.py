def get_manifest(self):
        """Return the manifest as a dictionary."""
        logger.debug("Getting manifest")
        text = self.get_text(self.get_manifest_key())
        return json.loads(text)