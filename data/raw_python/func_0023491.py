def check(self, remote_path=root):
        """Checks an existence of remote resource on WebDAV server by remote path.
        More information you can find by link http://webdav.org/specs/rfc4918.html#rfc.section.9.4

        :param remote_path: (optional) path to resource on WebDAV server. Defaults is root directory of WebDAV.
        :return: True if resource is exist or False otherwise
        """
        urn = Urn(remote_path)
        try:
            response = self.execute_request(action='check', path=urn.quote())
        except ResponseErrorCode:
            return False

        if int(response.status_code) == 200:
            return True

        return False