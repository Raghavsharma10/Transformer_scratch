def name(self):
        """Set the name of the current MOC.

        The new name should be given after this option.

        ::

            pymoctool ... --name 'New MOC name' --output new_moc.fits
        """

        if self.moc is None:
            self.moc = MOC()

        self.moc.name = self.params.pop()