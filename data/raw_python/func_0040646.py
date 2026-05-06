def getRegexpsByName(regexpNames = ['all']):
    ''' 
        Method that recovers the names of the <RegexpObject> in a given list.

        :param regexpNames:    list of strings containing the possible regexp.

        :return:    Array of <RegexpObject> classes.
    '''

    allRegexpList = getAllRegexp()
    if 'all' in regexpNames:
        return allRegexpList

    regexpList = []
    # going through the regexpList 
    for name in regexpNames:
        for r in allRegexpList:
            if name == r.name:
                regexpList.append(r)
    return regexpList