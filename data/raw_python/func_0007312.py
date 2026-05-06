def upload_file(self, filepath, overwrite=True):
        """Uploads a file to the temporary sauce storage."""
        method = 'POST'
        filename = os.path.split(filepath)[1]
        endpoint = '/rest/v1/storage/{}/{}?overwrite={}'.format(
            self.client.sauce_username, filename, "true" if overwrite else "false")
        with open(filepath, 'rb') as filehandle:
            body = filehandle.read()
        return self.client.request(method, endpoint, body,
                                   content_type='application/octet-stream')