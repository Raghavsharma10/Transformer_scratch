def max_word_width(myDict):
    '''
        currd = {0:'AutoPauseSpeed', 125:'HRLimitLow', 6:'Activity'}
        max_wordwidth(currd)
    '''
    maxValueWidth = 0
    for each in myDict:
        eachValueWidth = myDict[each].__len__()
        if(eachValueWidth > maxValueWidth):
            maxValueWidth = eachValueWidth
    return(maxValueWidth)