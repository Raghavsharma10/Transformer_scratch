def getMugshot(self):
        """
        Return the L{Mugshot} associated with this L{Person}, or an unstored
        L{Mugshot} pointing at a placeholder mugshot image.
        """
        mugshot = self.store.findUnique(
            Mugshot, Mugshot.person == self, default=None)
        if mugshot is not None:
            return mugshot
        return Mugshot.placeholderForPerson(self)