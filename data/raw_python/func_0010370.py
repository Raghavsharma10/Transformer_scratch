def makeSequenceAbsolute(relVSequence, minV, maxV):
    '''
    Makes every value in a sequence absolute
    '''

    return [(value * (maxV - minV)) + minV for value in relVSequence]