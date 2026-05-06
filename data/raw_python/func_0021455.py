def canChiNgay(nn, tt, nnnn, duongLich=True, timeZone=7, thangNhuan=False):
    """Summary

    Args:
        nn (int): ngày
        tt (int): tháng
        nnnn (int): năm
        duongLich (bool, optional): True nếu là dương lịch, False âm lịch
        timeZone (int, optional): Múi giờ
        thangNhuan (bool, optional): Có phải là tháng nhuận không?

    Returns:
        TYPE: Description
    """
    if duongLich is False:
        [nn, tt, nnnn] = L2S(nn, tt, nnnn, thangNhuan, timeZone)
    jd = jdFromDate(nn, tt, nnnn)
    # print jd
    canNgay = (jd + 9) % 10 + 1
    chiNgay = (jd + 1) % 12 + 1
    return [canNgay, chiNgay]