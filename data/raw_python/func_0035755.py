def put_manifest(self, manifest):
        """Store the manifest."""
        logger.debug("Putting manifest")
        text = json.dumps(manifest, indent=2, sort_keys=True)
        key = self.get_manifest_key()
        self.put_text(key, text)