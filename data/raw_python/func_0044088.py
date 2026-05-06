def identifier(self):
        """Set the identifier of the current MOC.

        The new identifier should be given after this option.

        ::

            pymoctool ... --id 'New MOC identifier' --output new_moc.fits
        """

        if self.moc is None:
            self.moc = MOC()

        self.moc.id = self.params.pop()