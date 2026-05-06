def personByName(self, name):
        """
        Retrieve the L{Person} item for the given Q2Q address,
        creating it first if necessary.

        @type name: C{unicode}
        """
        return self.store.findOrCreate(Person, organizer=self, name=name)