def _make_line(self, uid, command=None):
        """
        Prepares an IRC line in Herald's format
        """
        if command:
            return ":".join(("HRLD", command, uid))
        else:
            return ":".join(("HRLD", uid))