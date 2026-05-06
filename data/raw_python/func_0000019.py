def crop(data, crinfo):
    """
    Crop the data.

    crop(data, crinfo)

    :param crinfo: min and max for each axis - [[minX, maxX], [minY, maxY], [minZ, maxZ]]

    """
    crinfo = fix_crinfo(crinfo)
    return data[
        __int_or_none(crinfo[0][0]) : __int_or_none(crinfo[0][1]),
        __int_or_none(crinfo[1][0]) : __int_or_none(crinfo[1][1]),
        __int_or_none(crinfo[2][0]) : __int_or_none(crinfo[2][1]),
    ]