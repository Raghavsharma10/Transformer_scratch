def textgridMorphDuration(fromTGFN, toTGFN):
    '''
    A convenience function.  Morphs interval durations of one tg to another.
    
    This assumes the two textgrids have the same number of segments.
    '''
    fromTG = tgio.openTextgrid(fromTGFN)
    toTG = tgio.openTextgrid(toTGFN)
    adjustedTG = tgio.Textgrid()

    for tierName in fromTG.tierNameList:
        fromTier = fromTG.tierDict[tierName]
        toTier = toTG.tierDict[tierName]
        adjustedTier = fromTier.morph(toTier)
        adjustedTG.addTier(adjustedTier)

    return adjustedTG