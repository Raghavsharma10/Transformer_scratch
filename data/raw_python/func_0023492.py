def mkdir(self, remote_path):
        """Makes new directory on WebDAV server.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_MKCOL

        :param remote_path: path to directory
        :return: True if request executed with code 200 or 201 and False otherwise.

        """
        directory_urn = Urn(remote_path, directory=True)

        try:
            response = self.execute_request(action='mkdir', path=directory_urn.quote())

            return response.status_code in (200, 201)
        except ResponseErrorCode as e:
            if e.code == 405:
                return True

            raise