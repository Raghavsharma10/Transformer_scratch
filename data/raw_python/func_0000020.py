def combinecrinfo(crinfo1, crinfo2):
    """
    Combine two crinfos. First used is crinfo1, second used is crinfo2.
    """
    crinfo1 = fix_crinfo(crinfo1)
    crinfo2 = fix_crinfo(crinfo2)

    crinfo = [
        [crinfo1[0][0] + crinfo2[0][0], crinfo1[0][0] + crinfo2[0][1]],
        [crinfo1[1][0] + crinfo2[1][0], crinfo1[1][0] + crinfo2[1][1]],
        [crinfo1[2][0] + crinfo2[2][0], crinfo1[2][0] + crinfo2[2][1]],
    ]

    return crinfo