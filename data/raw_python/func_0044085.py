def read_moc(self, filename):
        """Read a file into the current running MOC object.

        If the running MOC object has not yet been created, then
        it is created by reading the file, which will import the
        MOC metadata.  Otherwise the metadata are not imported.
        """

        if self.moc is None:
            self.moc = MOC(filename=filename)

        else:
            self.moc.read(filename)