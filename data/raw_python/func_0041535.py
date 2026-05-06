def everyonesAverage(year, badFormat, length):
    ''' creates list of weighted average results for everyone in year

    Arguments:
        year {int}
        badFormat {dict} -- candNumber : [results for candidate]
        length {int} -- length of each row in badFormat divided by 2


    returns:
        list -- weighted average results of everyone in year
    '''
    return [myGrades(year, cand, badFormat, length) for cand in list(badFormat.keys())[1:]]