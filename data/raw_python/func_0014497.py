def remove(self, path):
        """
        Remove the file at the given path.  This only works on files; for
        removing folders (directories), use L{rmdir}.

        @param path: path (absolute or relative) of the file to remove
        @type path: str

        @raise IOError: if the path refers to a folder (directory)
        """
        path = self._adjust_cwd(path)
        self._log(DEBUG, 'remove(%r)' % path)
        self._request(CMD_REMOVE, path)