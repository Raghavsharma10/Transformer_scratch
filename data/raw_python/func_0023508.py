def clean(self, remote_path):
        """Cleans (Deletes) a remote resource on WebDAV server. The name of method is not changed for back compatibility
        with original library.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_DELETE

        :param remote_path: the remote resource whisch will be deleted.
        """
        urn = Urn(remote_path)
        self.execute_request(action='clean', path=urn.quote())