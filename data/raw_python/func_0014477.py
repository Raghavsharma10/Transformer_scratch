def canonicalize(self, path):
        """
        Return the canonical form of a path on the server.  For example,
        if the server's home folder is C{/home/foo}, the path
        C{"../betty"} would be canonicalized to C{"/home/betty"}.  Note
        the obvious security issues: if you're serving files only from a
        specific folder, you probably don't want this method to reveal path
        names outside that folder.

        You may find the python methods in C{os.path} useful, especially
        C{os.path.normpath} and C{os.path.realpath}.

        The default implementation returns C{os.path.normpath('/' + path)}.
        """
        if os.path.isabs(path):
            out = os.path.normpath(path)
        else:
            out = os.path.normpath('/' + path)
        if sys.platform == 'win32':
            # on windows, normalize backslashes to sftp/posix format
            out = out.replace('\\', '/')
        return out