def estimateAbsoluteMagnitude(spectralType):
    """Uses the spectral type to lookup an approximate absolute magnitude for
    the star.
    """

    from .astroclasses import SpectralType

    specType = SpectralType(spectralType)

    if specType.classLetter == '':
        return np.nan
    elif specType.classNumber == '':
        specType.classNumber = 5  # approximation using mid magnitude value

    if specType.lumType == '':
        specType.lumType = 'V'  # assume main sequence

    LNum = LClassRef[specType.lumType]
    classNum = specType.classNumber
    classLet = specType.classLetter

    try:
        return absMagDict[classLet][classNum][LNum]
    # value not in table. Assume the number isn't there (Key p2.7, Ind p3+)
    except (KeyError, IndexError):
        try:
            classLookup = absMagDict[classLet]
            values = np.array(list(classLookup.values()))[
                :, LNum]  # only select the right L Type
            return np.interp(classNum, list(classLookup.keys()), values)
        except (KeyError, ValueError):
            return np.nan