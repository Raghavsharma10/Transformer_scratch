def chmod(self, mode):
        """
        Change the mode (permissions) of this file.  The permissions are
        unix-style and identical to those used by python's C{os.chmod}
        function.

        @param mode: new permissions
        @type mode: int
        """
        self.sftp._log(DEBUG, 'chmod(%s, %r)' % (hexlify(self.handle), mode))
        attr = SFTPAttributes()
        attr.st_mode = mode
        self.sftp._request(CMD_FSETSTAT, self.handle, attr)