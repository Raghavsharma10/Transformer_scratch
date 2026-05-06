def lstat(self, path):
        """
        Retrieve information about a file on the remote system, without
        following symbolic links (shortcuts).  This otherwise behaves exactly
        the same as L{stat}.

        @param path: the filename to stat
        @type path: str
        @return: an object containing attributes about the given file
        @rtype: SFTPAttributes
        """
        path = self._adjust_cwd(path)
        self._log(DEBUG, 'lstat(%r)' % path)
        t, msg = self._request(CMD_LSTAT, path)
        if t != CMD_ATTRS:
            raise SFTPError('Expected attributes')
        return SFTPAttributes._from_msg(msg)