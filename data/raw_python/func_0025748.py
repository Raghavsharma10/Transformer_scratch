def getExecuteStrings(self, parScope, parName):
        """ For a given item (scope + name), return all strings (in a tuple)
        that it is meant to execute, if any exist.  Returns None is none. """
        # The data structure of _allExecutes was chosen for how easily/quickly
        # this particular access can be made here.
        fullName = parScope+'.'+parName
        return self._allExecutes.get(fullName)