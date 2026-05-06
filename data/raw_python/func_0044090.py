def intersection(self):
        """Compute the intersection with the given MOC.

        This command takes the name of a MOC file and forms the intersection
        of the running MOC with that file.

        ::

            pymoctool a.fits --intersection b.fits --output intersection.fits
        """

        if self.moc is None:
            raise CommandError('No MOC information present for intersection')

        filename = self.params.pop()
        self.moc = self.moc.intersection(MOC(filename=filename))