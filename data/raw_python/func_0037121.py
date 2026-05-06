def get_processed_key_name(self):
        """Return the full path to use for the processed file."""
        if not hasattr(self, '_processed_key_name'):
            path, upload_name = os.path.split(self.get_upload_key().name)
            key_name = self._generate_processed_key_name(
                self.process_to, upload_name)
            self._processed_key_name = os.path.join(
                self.get_storage().location, key_name)
        return self._processed_key_name