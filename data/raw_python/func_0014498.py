def rmdir(self, path):
        """
        Remove the folder named C{path}.

        @param path: name of the folder to remove
        @type path: str
        """
        path = self._adjust_cwd(path)
        self._log(DEBUG, 'rmdir(%r)' % path)
        self._request(CMD_RMDIR, path)