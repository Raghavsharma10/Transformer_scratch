def normalize(self):
        """Normalize the MOC to a given order.

        This command takes a MOC order (0-29) and normalizes the MOC so that
        its maximum order is the given order.

        ::

            pymoctool a.fits --normalize 10 --output a_10.fits
        """

        if self.moc is None:
            raise CommandError('No MOC information present for normalization')

        order = int(self.params.pop())
        self.moc.normalize(order)