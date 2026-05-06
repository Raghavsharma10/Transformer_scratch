def getTriggerStrings(self, parScope, parName):
        """ For a given item (scope + name), return all strings (in a tuple)
        that it is meant to trigger, if any exist.  Returns None is none. """
        # The data structure of _allTriggers was chosen for how easily/quickly
        # this particular access can be made here.
        fullName = parScope+'.'+parName
        return self._allTriggers.get(fullName)