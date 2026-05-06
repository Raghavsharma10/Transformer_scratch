def _adjust_cwd(self, path):
        """
        Return an adjusted path if we're emulating a "current working
        directory" for the server.
        """
        if type(path) is unicode:
            path = path.encode('utf-8')
        if self._cwd is None:
            return path
        if (len(path) > 0) and (path[0] == '/'):
            # absolute path
            return path
        if self._cwd == '/':
            return self._cwd + path
        return self._cwd + '/' + path