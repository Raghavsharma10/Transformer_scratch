def upload_waf_log(self, waf_id, file_path):
        """
        Uploads and processes a WAF log.
        :param waf_id: WAF identifier.
        :param file_path: Path to the WAF log file to be uploaded.
        """
        return self._request('POST', 'rest/wafs/' + str(waf_id) + '/uploadLog', files={'file': open(file_path, 'rb')})