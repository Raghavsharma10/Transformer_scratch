def setParam(self, name, val, scope='', check=1, idxHint=None):
        """ Find the ConfigObj entry.  Update the __paramList. """
        theDict, oldVal = findScopedPar(self, scope, name)

        # Set the value, even if invalid.  It needs to be set before
        # the validation step (next).
        theDict[name] = val

        # If need be, check the proposed value.  Ideally, we'd like to
        # (somehow elegantly) only check this one item. For now, the best
        # shortcut is to only validate this section.
        if check:
            ans=self.validate(self._vtor, preserve_errors=True, section=theDict)
            if ans != True:
                flatStr = "All values are invalid!"
                if ans != False:
                    flatStr = flattened2str(configobj.flatten_errors(self, ans))
                raise RuntimeError("Validation error: "+flatStr)

        # Note - this design needs work.  Right now there are two copies
        # of the data:  the ConfigObj dict, and the __paramList ...
        # We rely on the idxHint arg so we don't have to search the __paramList
        # every time this is called, which could really slows things down.
        assert idxHint is not None, "ConfigObjPars relies on a valid idxHint"
        assert name == self.__paramList[idxHint].name, \
               'Error in setParam, name: "'+name+'" != name at idxHint: "'+\
               self.__paramList[idxHint].name+'", idxHint: '+str(idxHint)
        self.__paramList[idxHint].set(val)