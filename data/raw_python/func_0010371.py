def _makeTimingRelative(absoluteDataList):
    '''
    Given normal pitch tier data, puts the times on a scale from 0 to 1

    Input is a list of tuples of the form
    ([(time1, pitch1), (time2, pitch2),...]

    Also returns the start and end time so that the process can be reversed
    '''

    timingSeq = [row[0] for row in absoluteDataList]
    valueSeq = [list(row[1:]) for row in absoluteDataList]

    relTimingSeq, startTime, endTime = makeSequenceRelative(timingSeq)
    
    relDataList = [tuple([time, ] + row) for time, row
                   in zip(relTimingSeq, valueSeq)]

    return relDataList, startTime, endTime