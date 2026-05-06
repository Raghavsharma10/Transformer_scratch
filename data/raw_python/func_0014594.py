def upload(self, src_file_path, dst_file_name=None):
        """Upload the specified file to the server."""
        self._check_session()
        status, data = self._rest.upload_file(
            'files', src_file_path, dst_file_name)
        return data