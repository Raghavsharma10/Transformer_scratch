def ngayThangNam(nn, tt, nnnn, duongLich=True, timeZone=7):
    """Summary

    Args:
        nn (TYPE): ngay
        tt (TYPE): thang
        nnnn (TYPE): nam
        duongLich (bool, optional): bool
        timeZone (int, optional): +7 Vietnam

    Returns:
        TYPE: Description

    Raises:
        Exception: Description
    """
    thangNhuan = 0
    # if nnnn > 1000 and nnnn < 3000 and nn > 0 and \
    if nn > 0 and \
       nn < 32 and tt < 13 and tt > 0:
        if duongLich is True:
            [nn, tt, nnnn, thangNhuan] = S2L(nn, tt, nnnn, timeZone=timeZone)
        return [nn, tt, nnnn, thangNhuan]
    else:
        raise Exception("Ngày, tháng, năm không chính xác.")