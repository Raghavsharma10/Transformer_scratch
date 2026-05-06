def write(self, *a, **kw):
        """
        Write to both files

        If either one has an error, try writing the error to the other one.
        """
        fl = None
        try:
            self.file1.write(*a, **kw)
            self.file1.flush()
        except IOError:
            badFile, fl = 1, failure.Failure()

        try:
            self.file2.write(*a, **kw)
            self.file2.flush()
        except IOError:
            badFile, fl = 2, failure.Failure()

        if fl:
            out = self.file2 if badFile == 1 else self.file1
            out.write(str(fl) + '\n')
            out.flush()
            fl.raiseException()