def getInitialLiveForms(self):
        """
        Make and return as many L{LiveForm} instances as are necessary to hold
        our default values.

        @return: some subforms.
        @rtype: C{list} of L{LiveForm}
        """
        liveForms = []
        if self._defaultStuff:
            for values in self._defaultStuff:
                liveForms.append(self._makeDefaultLiveForm(values))
        else:
            # or only one, for the first new thing
            liveForms.append(
                self._makeALiveForm(
                    self.parameters, self._newIdentifier(), False))
        return liveForms