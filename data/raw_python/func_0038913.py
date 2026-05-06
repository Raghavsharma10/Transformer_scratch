def checksums(self, path, hashtype='sha1'):
        """Iterates over the files hashes contained within the disk
        starting from the given path.

        The hashtype keyword allows to choose the file hashing algorithm.

        Yields the following values:

            "C:\\Windows\\System32\\NTUSER.DAT", "hash" for windows
            "/home/user/text.txt", "hash" for other FS

        """
        with NamedTemporaryFile(buffering=0) as tempfile:
            self._handler.checksums_out(hashtype, posix_path(path),
                                        tempfile.name)

            yield from ((self.path(f[1].lstrip('.')), f[0])
                        for f in (l.decode('utf8').strip().split(None, 1)
                                  for l in tempfile))