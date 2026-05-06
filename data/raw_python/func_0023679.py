def upload_file(self, source, dest_uri):
        """Upload file to MediaFire.

        source -- path to the file or a file-like object (e.g. io.BytesIO)
        dest_uri -- MediaFire Resource URI
        """

        folder_key, name = self._prepare_upload_info(source, dest_uri)

        is_fh = hasattr(source, 'read')
        fd = None

        try:
            if is_fh:
                # Re-using filehandle
                fd = source
            else:
                # Handling fs open/close
                fd = open(source, 'rb')

            return MediaFireUploader(self.api).upload(
                fd, name, folder_key=folder_key,
                action_on_duplicate='replace')
        finally:
            # Close filehandle if we opened it
            if fd and not is_fh:
                fd.close()