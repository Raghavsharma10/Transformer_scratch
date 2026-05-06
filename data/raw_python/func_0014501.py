def symlink(self, source, dest):
        """
        Create a symbolic link (shortcut) of the C{source} path at
        C{destination}.

        @param source: path of the original file
        @type source: str
        @param dest: path of the newly created symlink
        @type dest: str
        """
        dest = self._adjust_cwd(dest)
        self._log(DEBUG, 'symlink(%r, %r)' % (source, dest))
        if type(source) is unicode:
            source = source.encode('utf-8')
        self._request(CMD_SYMLINK, source, dest)