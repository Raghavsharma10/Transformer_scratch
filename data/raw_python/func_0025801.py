def _setupDefaultParamList(self):
        """ This creates self.defaultParamList.  It also does some checks
        on the paramList, sets its order if needed, and deletes any extra
        or unknown pars if found. We assume the order of self.defaultParamList
        is the correct order. """

        # Obtain the default parameter list
        self.defaultParamList = self._taskParsObj.getDefaultParList()
        theParamList = self._taskParsObj.getParList()

        # Lengths are probably equal but this isn't necessarily an error
        # here, so we check for differences below.
        if len(self.defaultParamList) != len(theParamList):
            # whoa, lengths don't match (could be some missing or some extra)
            pmsg = 'Current list not same length as default list'
            if not self._handleParListMismatch(pmsg):
                return False

        # convert current par values to a dict of { par-fullname:par-object }
        # for use below
        ourpardict = {}
        for par in theParamList: ourpardict[par.fullName()] = par

        # Sort our paramList according to the order of the defaultParamList
        # and repopulate the list according to that order. Create sortednames.
        sortednames = [p.fullName() for p in self.defaultParamList]

        # Rebuild par list sorted into correct order.  Also find/flag any
        # missing pars or any extra/unknown pars.  This automatically deletes
        # "extras" by not adding them to the sorted list in the first place.
        migrated = []
        newList = []
        for fullName in sortednames:
            if fullName in ourpardict:
                newList.append(ourpardict[fullName])
                migrated.append(fullName) # make sure all get moved over
            else: # this is a missing par - insert the default version
                theDfltVer = \
                    [p for p in self.defaultParamList if p.fullName()==fullName]
                newList.append(copy.deepcopy(theDfltVer[0]))

        # Update!  Next line writes to the self._taskParsObj.getParList() obj
        theParamList[:] = newList # fill with newList, keep same mem pointer

        # See if any got left out
        extras = [fn for fn in ourpardict if not fn in migrated]
        for fullName in extras:
            # this is an extra/unknown par - let subclass handle it
            if not self._handleParListMismatch('Unexpected par: "'+\
                        fullName+'"', extra=True):
                return False
            print('Ignoring unexpected par: "'+p+'"')

        # return value indicates that all is well to continue
        return True