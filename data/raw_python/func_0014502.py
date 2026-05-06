def chmod(self, path, mode):
        """
        Change the mode (permissions) of a file.  The permissions are
        unix-style and identical to those used by python's C{os.chmod}
        function.

        @param path: path of the file to change the permissions of
        @type path: str
        @param mode: new permissions
        @type mode: int
        """
        path = self._adjust_cwd(path)
        self._log(DEBUG, 'chmod(%r, %r)' % (path, mode))
        attr = SFTPAttributes()
        attr.st_mode = mode
        self._request(CMD_SETSTAT, path, attr)