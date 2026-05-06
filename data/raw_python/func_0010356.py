def getPitchForIntervals(data, tgFN, tierName):
    '''
    Preps data for use in f0Morph
    '''
    tg = tgio.openTextgrid(tgFN)
    data = tg.tierDict[tierName].getValuesInIntervals(data)
    data = [dataList for _, dataList in data]

    return data