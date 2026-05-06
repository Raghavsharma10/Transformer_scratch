def copy(self, remote_path_from, remote_path_to):
        """Copies resource from one place to another on WebDAV server.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_COPY

        :param remote_path_from: the path to resource which will be copied,
        :param remote_path_to: the path where resource will be copied.
        """
        urn_from = Urn(remote_path_from)
        if not self.check(urn_from.path()):
            raise RemoteResourceNotFound(urn_from.path())

        urn_to = Urn(remote_path_to)
        if not self.check(urn_to.parent()):
            raise RemoteParentNotFound(urn_to.path())

        header_destination = f'Destination: {self.get_full_path(urn_to)}'
        self.execute_request(action='copy', path=urn_from.quote(), headers_ext=[header_destination])