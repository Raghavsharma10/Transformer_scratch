def getInitialArguments(self):
        """
        Include L{organizer}'s C{storeOwnerPerson}'s name, and the name of
        L{initialPerson} and the value of L{initialState}, if they are set.
        """
        initialArguments = (self.organizer.storeOwnerPerson.name,)
        if self.initialPerson is not None:
            initialArguments += (self.initialPerson.name, self.initialState)
        return initialArguments