def getIntervals(fn, tierName, filterFunc=None,
                 includeUnlabeledRegions=False):
    '''
    Get information about the 'extract' tier, used by several merge scripts
    '''

    tg = tgio.openTextgrid(fn)
    
    tier = tg.tierDict[tierName]
    if includeUnlabeledRegions is True:
        tier = tgio._fillInBlanks(tier)

    entryList = tier.entryList
    if filterFunc is not None:
        entryList = [entry for entry in entryList if filterFunc(entry)]

    return entryList