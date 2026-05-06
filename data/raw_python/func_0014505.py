def get(self, remotepath, localpath, callback=None):
        """
        Copy a remote file (C{remotepath}) from the SFTP server to the local
        host as C{localpath}.  Any exception raised by operations will be
        passed through.  This method is primarily provided as a convenience.

        @param remotepath: the remote file to copy
        @type remotepath: str
        @param localpath: the destination path on the local host
        @type localpath: str
        @param callback: optional callback function that accepts the bytes
            transferred so far and the total bytes to be transferred
            (since 1.7.4)
        @type callback: function(int, int)

        @since: 1.4
        """
        fr = self.file(remotepath, 'rb')
        file_size = self.stat(remotepath).st_size
        fr.prefetch()
        try:
            fl = file(localpath, 'wb')
            try:
                size = 0
                while True:
                    data = fr.read(32768)
                    if len(data) == 0:
                        break
                    fl.write(data)
                    size += len(data)
                    if callback is not None:
                        callback(size, file_size)
            finally:
                fl.close()
        finally:
            fr.close()
        s = os.stat(localpath)
        if s.st_size != size:
            raise IOError('size mismatch in get!  %d != %d' % (s.st_size, size))