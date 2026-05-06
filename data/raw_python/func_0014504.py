def normalize(self, path):
        """
        Return the normalized path (on the server) of a given path.  This
        can be used to quickly resolve symbolic links or determine what the
        server is considering to be the "current folder" (by passing C{'.'}
        as C{path}).

        @param path: path to be normalized
        @type path: str
        @return: normalized form of the given path
        @rtype: str

        @raise IOError: if the path can't be resolved on the server
        """
        path = self._adjust_cwd(path)
        self._log(DEBUG, 'normalize(%r)' % path)
        t, msg = self._request(CMD_REALPATH, path)
        if t != CMD_NAME:
            raise SFTPError('Expected name response')
        count = msg.get_int()
        if count != 1:
            raise SFTPError('Realpath returned %d results' % count)
        return _to_unicode(msg.get_string())