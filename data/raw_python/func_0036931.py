def max_display_width(myDict):
    '''
        currd = {0:'你们大家好', 125:'ABCDE', 6:'1234567'}
        dict_get_max_word_displaywidth(currd)
    '''
    maxValueWidth = 0
    for each in myDict:
        eachValueWidth = str_display_width(myDict[each])
        if(eachValueWidth > maxValueWidth):
            maxValueWidth = eachValueWidth
    return(maxValueWidth)