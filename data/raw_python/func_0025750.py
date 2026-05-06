def getKwdArgs(self, flatten = False):
        """ Return a dict of all normal dict parameters - that is, all
            parameters NOT marked with "pos=N" in the .cfgspc file.  This will
            also exclude all hidden parameters (metadata, rules, etc). """

        # Start with a full deep-copy.  What complicates this method is the
        # idea of sub-sections.  This dict can have dicts as values, and so on.
        dcopy = self.dict() # ConfigObj docs say this is a deep-copy

        # First go through the dict removing all positional args
        for idx,scope,name in self._posArgs:
            theDict, val = findScopedPar(dcopy, scope, name)
            # 'theDict' may be dcopy, or it may be a dict under it
            theDict.pop(name)

        # Then go through the dict removing all hidden items ('_item_name_')
        for k in list(dcopy.keys()):
            if isHiddenName(k):
                dcopy.pop(k)

        # Done with the nominal operation
        if not flatten:
            return dcopy

        # They have asked us to flatten the structure - to bring all parameters
        # up to the top level, even if they are in sub-sections.  So we look
        # for values that are dicts.  We will throw something if we end up
        # with name collisions at the top level as a result of this.
        return flattenDictTree(dcopy)