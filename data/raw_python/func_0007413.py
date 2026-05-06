def get_stat_json(self, pathobj):
        """
        Request remote file/directory status info
        Returns a json object as specified by Artifactory REST API
        """
        url = '/'.join([pathobj.drive,
                        'api/storage',
                        str(pathobj.relative_to(pathobj.drive)).strip('/')])

        text, code = self.rest_get(url, auth=pathobj.auth, verify=pathobj.verify,
                                   cert=pathobj.cert)
        if code == 404 and "Unable to find item" in text:
            raise OSError(2, "No such file or directory: '%s'" % url)
        if code != 200:
            raise RuntimeError(text)

        return json.loads(text)