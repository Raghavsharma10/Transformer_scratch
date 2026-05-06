def getContactTypes(self):
        """
        Return an iterator of L{IContactType} providers available to this
        organizer's store.
        """
        yield VIPPersonContactType()
        yield EmailContactType(self.store)
        yield PostalContactType()
        yield PhoneNumberContactType()
        yield NotesContactType()
        for getContactTypes in self._gatherPluginMethods('getContactTypes'):
            for contactType in getContactTypes():
                self._checkContactType(contactType)
                yield contactType