def move(self, remote_path_from, remote_path_to, overwrite=False):
        """Moves resource from one place to another on WebDAV server.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_MOVE

        :param remote_path_from: the path to resource which will be moved,
        :param remote_path_to: the path where resource will be moved.
        :param overwrite: (optional) the flag, overwrite file if it exists. Defaults is False
        """
        urn_from = Urn(remote_path_from)
        if not self.check(urn_from.path()):
            raise RemoteResourceNotFound(urn_from.path())

        urn_to = Urn(remote_path_to)
        if not self.check(urn_to.parent()):
            raise RemoteParentNotFound(urn_to.path())

        header_destination = f'Destination: {self.get_full_path(urn_to)}'
        header_overwrite = f'Overwrite: {"T" if overwrite else "F"}'
        self.execute_request(action='move', path=urn_from.quote(), headers_ext=[header_destination, header_overwrite])