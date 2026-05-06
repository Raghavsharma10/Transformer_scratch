def ngayThangNamCanChi(nn, tt, nnnn, duongLich=True, timeZone=7):
    """chuyển đổi năm, tháng âm/dương lịch sang Can, Chi trong tiếng Việt.
    Không tính đến can ngày vì phải chuyển đổi qua lịch Julius.

    Hàm tìm can ngày là hàm canChiNgay(nn, tt, nnnn, duongLich=True,\
                                    timeZone=7, thangNhuan=False)

    Args:
        nn (int): Ngày
        tt (int): Tháng
        nnnn (int): Năm

    Returns:
        TYPE: Description
    """
    if duongLich is True:
        [nn, tt, nnnn, thangNhuan] = \
            ngayThangNam(nn, tt, nnnn, timeZone=timeZone)
    # Can của tháng
    canThang = (nnnn * 12 + tt + 3) % 10 + 1
    # Can chi của năm
    canNamSinh = (nnnn + 6) % 10 + 1
    chiNam = (nnnn + 8) % 12 + 1

    return [canThang, canNamSinh, chiNam]