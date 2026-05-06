def findNextSection(self, scope, name):
        """ Starts with given par (scope+name) and looks further down the list
        of parameters until one of a different non-null scope is found.  Upon
        success, returns the (scope, name) tuple, otherwise (None, None). """
        # first find index of starting point
        plist = self._taskParsObj.getParList()
        start = 0
        for i in range(len(plist)):
            if scope == plist[i].scope and name == plist[i].name:
                start = i
                break
        else:
            print('WARNING: could not find starting par: '+scope+'.'+name)
            return (None, None)

        # now find first different (non-null) scope in a par, after start
        for i in range(start, len(plist)):
            if len(plist[i].scope) > 0 and plist[i].scope != scope:
                return (plist[i].scope, plist[i].name)
        # else didn't find it
        return (None, None)