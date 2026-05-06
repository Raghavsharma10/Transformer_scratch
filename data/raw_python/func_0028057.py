def set_file_to_upload(self, file_to_upload):
        # type: (str) -> None
        """Delete any existing url and set the file uploaded to the local path provided

        Args:
            file_to_upload (str): Local path to file to upload

        Returns:
            None
        """
        if 'url' in self.data:
            del self.data['url']
        self.file_to_upload = file_to_upload