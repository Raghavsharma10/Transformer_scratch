def makeSequenceRelative(absVSequence):
    '''
    Puts every value in a list on a continuum between 0 and 1

    Also returns the min and max values (to reverse the process)
    '''

    if len(absVSequence) < 2 or len(set(absVSequence)) == 1:
        raise RelativizeSequenceException(absVSequence)

    minV = min(absVSequence)
    maxV = max(absVSequence)
    relativeSeq = [(value - minV) / (maxV - minV) for value in absVSequence]

    return relativeSeq, minV, maxV