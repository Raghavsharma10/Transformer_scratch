def upload_scan(self, application_id, file_path):
        """
        Uploads and processes a scan file.
        :param application_id: Application identifier.
        :param file_path: Path to the scan file to be uploaded.
        """
        return self._request(
            'POST', 'rest/applications/' + str(application_id) + '/upload',
            files={'file': open(file_path, 'rb')}
        )