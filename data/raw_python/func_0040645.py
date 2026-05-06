def getAllRegexpNames(regexpList = None):
    ''' 
        Method that recovers the names of the <RegexpObject> in a given list.

        :param regexpList:    list of <RegexpObject>. If None, all the available <RegexpObject> will be recovered.

        :return:    Array of strings containing the available regexps.
    '''
    if regexpList == None:
        regexpList = getAllRegexp()
    listNames = ['all']
    # going through the regexpList 
    for r in regexpList:
        listNames.append(r.name)
    return listNames