def myGrades(year, candidateNumber, badFormat, length):
    '''returns final result of candidateNumber in year

    Arguments:
        year {int} -- the year candidateNumber is in
        candidateNumber {str} -- the candidateNumber of candidateNumber
        badFormat {dict} -- candNumber : [results for candidate]
        length {int} -- length of each row in badFormat divided by 2


    Returns:
        int -- a weighted average for a specific candidate number and year
    '''

    weights1 = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
    weights2 = [1, 1, 1, 1, 1, 1, 0.5, 0.5]
    if year == 1:
        myFinalResult = sum([int(badFormat[candidateNumber][2*(i + 1)])
                             * weights1[i] for i in range(length-1)]) / 6
    elif year == 2 or year == 3:
        myFinalResult = sum([int(badFormat[candidateNumber][2*(i + 1)])
                             * weights2[i] for i in range(length-1)]) / 7
    elif year == 4:
        myFinalResult = sum([int(badFormat[candidateNumber][2*(i + 1)])
                             for i in range(length-1)]) / 8

    return myFinalResult