def getValue(self, name, scope=None, native=False):
        """ Return current par value from the GUI. This does not do any
        validation, and it it not necessarily the same value saved in the
        model, which is always behind the GUI setting, in time. This is NOT
        to be used to get all the values - it would not be efficient. """

        # Get model data, the list of pars
        theParamList = self._taskParsObj.getParList()

        # NOTE: If par scope is given, it will be used, otherwise it is
        # assumed to be unneeded and the first name-match is returned.
        fullName = basicpar.makeFullName(scope, name)

        # Loop over the parameters to find the requested par
        for i in range(self.numParams):
            par = theParamList[i] # IrafPar or subclass
            entry = self.entryNo[i] # EparOption or subclass
            if par.fullName() == fullName or \
               (scope is None and par.name == name):
                if native:
                    return entry.convertToNative(entry.choice.get())
                else:
                    return entry.choice.get()
        # We didn't find the requested par
        raise RuntimeError('Could not find par: "'+fullName+'"')