def readlink(self, path):
        """
        Return the target of a symbolic link (shortcut).  You can use
        L{symlink} to create these.  The result may be either an absolute or
        relative pathname.

        @param path: path of the symbolic link file
        @type path: str
        @return: target path
        @rtype: str
        """
        path = self._adjust_cwd(path)
        self._log(DEBUG, 'readlink(%r)' % path)
        t, msg = self._request(CMD_READLINK, path)
        if t != CMD_NAME:
            raise SFTPError('Expected name response')
        count = msg.get_int()
        if count == 0:
            return None
        if count != 1:
            raise SFTPError('Readlink returned %d results' % count)
        return _to_unicode(msg.get_string())