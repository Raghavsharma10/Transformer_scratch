def tryValue(self, name, val, scope=''):
        """ For the given item name (and scope), we are being asked to try
            the given value to see if it would pass validation.  We are not
            to set it, but just try it.  We return a tuple:
            If it fails, we return: (False,  the last known valid value).
            On success, we return: (True, None). """

        # SIMILARITY BETWEEN THIS AND setParam() SHOULD BE CONSOLIDATED!

        # Set the value, even if invalid.  It needs to be set before
        # the validation step (next).
        theDict, oldVal = findScopedPar(self, scope, name)
        if oldVal == val: return (True, None) # assume oldVal is valid
        theDict[name] = val

        # Check the proposed value.  Ideally, we'd like to
        # (somehow elegantly) only check this one item. For now, the best
        # shortcut is to only validate this section.
        ans=self.validate(self._vtor, preserve_errors=True, section=theDict)

        # No matter what ans is, immediately return the item to its original
        # value since we are only checking the value here - not setting.
        theDict[name] = oldVal

        # Now see what the validation check said
        errStr = ''
        if ans != True:
            flatStr = "All values are invalid!"
            if ans != False:
                flatStr = flattened2str(configobj.flatten_errors(self, ans))
            errStr = "Validation error: "+flatStr # for now this info is unused

        # Done
        if len(errStr): return (False, oldVal) # was an error
        else:           return (True, None)