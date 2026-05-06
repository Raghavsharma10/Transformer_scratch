def subtract(self):
        """Subtract the given MOC from the running MOC.

        This command takes the name of a MOC file to be subtracted from the
        running MOC.

        ::

            pymoctool a.fits --subtract b.fits --output difference.fits
        """

        if self.moc is None:
            raise CommandError('No MOC information present for subtraction')

        filename = self.params.pop()
        self.moc -= MOC(filename=filename)