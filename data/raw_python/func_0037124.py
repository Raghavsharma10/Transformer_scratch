def get_upload_content_type(self):
        """Determine the actual content type of the upload."""
        if not hasattr(self, '_upload_content_type'):
            with self.get_storage().open(self.get_upload_path()) as upload:
                content_type = Magic(mime=True).from_buffer(upload.read(1024))
            self._upload_content_type = content_type
        return self._upload_content_type