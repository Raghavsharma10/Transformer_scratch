def timTuVi(cuc, ngaySinhAmLich):
    """Tìm vị trí của sao Tử vi

    Args:
        cuc (TYPE): Description
        ngaySinhAmLich (TYPE): Description

    Returns:
        TYPE: Description

    Raises:
        Exception: Description
    """
    cungDan = 3  # Vị trí cung Dần ban đầu là 3
    cucBanDau = cuc
    if cuc not in [2, 3, 4, 5, 6]:  # Tránh trường hợp infinite loop
        raise Exception("Số cục phải là 2, 3, 4, 5, 6")
    while cuc < ngaySinhAmLich:
        cuc += cucBanDau
        cungDan += 1  # Dịch vị trí cung Dần
    saiLech = cuc - ngaySinhAmLich
    if saiLech % 2 is 1:
        saiLech = -saiLech  # Nếu sai lệch là chẵn thì tiến, lẻ thì lùi
    return dichCung(cungDan, saiLech)