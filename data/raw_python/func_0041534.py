def myRank(grade, badFormat, year, length):
    '''rank of candidateNumber in year

    Arguments:
        grade {int} -- a weighted average for a specific candidate number and year
        badFormat {dict} -- candNumber : [results for candidate]
        year {int} -- year you are in
        length {int} -- length of each row in badFormat divided by 2



    Returns:
        int -- rank of candidateNumber in year
    '''
    return int(sorted(everyonesAverage(year, badFormat, length), reverse=True).index(grade) + 1)