def write_moc(self):
        """Write the MOC to a given file."""

        if self.moc is None:
            raise CommandError('No MOC information present for output')

        filename = self.params.pop()
        self.moc.write(filename)