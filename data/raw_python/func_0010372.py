def _makeTimingAbsolute(relativeDataList, startTime, endTime):
    '''
    Maps values from 0 to 1 to the provided start and end time

    Input is a list of tuples of the form
    ([(time1, pitch1), (time2, pitch2),...]
    '''

    timingSeq = [row[0] for row in relativeDataList]
    valueSeq = [list(row[1:]) for row in relativeDataList]
    
    absTimingSeq = makeSequenceAbsolute(timingSeq, startTime, endTime)

    absDataList = [tuple([time, ] + row) for time, row
                   in zip(absTimingSeq, valueSeq)]

    return absDataList