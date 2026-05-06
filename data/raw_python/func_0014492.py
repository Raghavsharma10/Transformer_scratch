def utime(self, times):
        """
        Set the access and modified times of this file.  If
        C{times} is C{None}, then the file's access and modified times are set
        to the current time.  Otherwise, C{times} must be a 2-tuple of numbers,
        of the form C{(atime, mtime)}, which is used to set the access and
        modified times, respectively.  This bizarre API is mimicked from python
        for the sake of consistency -- I apologize.

        @param times: C{None} or a tuple of (access time, modified time) in
            standard internet epoch time (seconds since 01 January 1970 GMT)
        @type times: tuple(int)
        """
        if times is None:
            times = (time.time(), time.time())
        self.sftp._log(DEBUG, 'utime(%s, %r)' % (hexlify(self.handle), times))
        attr = SFTPAttributes()
        attr.st_atime, attr.st_mtime = times
        self.sftp._request(CMD_FSETSTAT, self.handle, attr)