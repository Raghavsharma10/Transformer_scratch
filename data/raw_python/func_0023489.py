def list(self, remote_path=root):
        """Returns list of nested files and directories for remote WebDAV directory by path.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PROPFIND

        :param remote_path: path to remote directory.
        :return: list of nested file or directory names.
        """
        directory_urn = Urn(remote_path, directory=True)
        if directory_urn.path() != Client.root:
            if not self.check(directory_urn.path()):
                raise RemoteResourceNotFound(directory_urn.path())

        response = self.execute_request(action='list', path=directory_urn.quote())
        urns = WebDavXmlUtils.parse_get_list_response(response.content)

        path = Urn.normalize_path(self.get_full_path(directory_urn))
        return [urn.filename() for urn in urns if Urn.compare_path(path, urn.path()) is False]