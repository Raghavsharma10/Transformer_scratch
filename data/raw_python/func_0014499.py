def stat(self, path):
        """
        Retrieve information about a file on the remote system.  The return
        value is an object whose attributes correspond to the attributes of
        python's C{stat} structure as returned by C{os.stat}, except that it
        contains fewer fields.  An SFTP server may return as much or as little
        info as it wants, so the results may vary from server to server.

        Unlike a python C{stat} object, the result may not be accessed as a
        tuple.  This is mostly due to the author's slack factor.

        The fields supported are: C{st_mode}, C{st_size}, C{st_uid}, C{st_gid},
        C{st_atime}, and C{st_mtime}.

        @param path: the filename to stat
        @type path: str
        @return: an object containing attributes about the given file
        @rtype: SFTPAttributes
        """
        path = self._adjust_cwd(path)
        self._log(DEBUG, 'stat(%r)' % path)
        t, msg = self._request(CMD_STAT, path)
        if t != CMD_ATTRS:
            raise SFTPError('Expected attributes')
        return SFTPAttributes._from_msg(msg)