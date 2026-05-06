def close(self):
        """
        Close this file and commit it to its permanent location.

        @return: a Deferred which fires when the file has been moved (and
        backed up to tertiary storage, if necessary).
        """
        now = time.time()
        try:
            file.close(self)
            _mkdirIfNotExists(self._destpath.dirname())
            self.finalpath = self._destpath
            os.rename(self.name, self.finalpath.path)
            os.utime(self.finalpath.path, (now, now))
        except:
            return defer.fail()
        return defer.succeed(self.finalpath)