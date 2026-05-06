def chown(self, uid, gid):
        """
        Change the owner (C{uid}) and group (C{gid}) of this file.  As with
        python's C{os.chown} function, you must pass both arguments, so if you
        only want to change one, use L{stat} first to retrieve the current
        owner and group.

        @param uid: new owner's uid
        @type uid: int
        @param gid: new group id
        @type gid: int
        """
        self.sftp._log(DEBUG, 'chown(%s, %r, %r)' % (hexlify(self.handle), uid, gid))
        attr = SFTPAttributes()
        attr.st_uid, attr.st_gid = uid, gid
        self.sftp._request(CMD_FSETSTAT, self.handle, attr)