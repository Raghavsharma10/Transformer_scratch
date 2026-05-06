def nguHanhNapAm(diaChi, thienCan, xuatBanMenh=False):
    """Sử dụng Ngũ Hành nạp âm để tính Hành của năm.

    Args:
        diaChi (integer): Số thứ tự của địa chi (Tý=1, Sửu=2,...)
        thienCan (integer): Số thứ tự của thiên can (Giáp=1, Ất=2,...)

    Returns:
        Trả về chữ viết tắt Hành của năm (K, T, H, O, M)
    """
    banMenh = {
        "K1": "HẢI TRUNG KIM",
        "T1": "GIÁNG HẠ THỦY",
        "H1": "TÍCH LỊCH HỎA",
        "O1": "BÍCH THƯỢNG THỔ",
        "M1": "TANG ÐỐ MỘC",
        "T2": "ÐẠI KHÊ THỦY",
        "H2": "LƯ TRUNG HỎA",
        "O2": "THÀNH ÐẦU THỔ",
        "M2": "TÒNG BÁ MỘC",
        "K2": "KIM BẠCH KIM",
        "H3": "PHÚ ÐĂNG HỎA",
        "O3": "SA TRUNG THỔ",
        "M3": "ÐẠI LÂM MỘC",
        "K3": "BẠCH LẠP KIM",
        "T3": "TRƯỜNG LƯU THỦY",
        "K4": "SA TRUNG KIM",
        "T4": "THIÊN HÀ THỦY",
        "H4": "THIÊN THƯỢNG HỎA",
        "O4": "LỘ BÀN THỔ",
        "M4": "DƯƠNG LIỄU MỘC",
        "T5": "TRUYỀN TRUNG THỦY",
        "H5": "SƠN HẠ HỎA",
        "O5": "ÐẠI TRẠCH THỔ",
        "M5": "THẠCH LỰU MỘC",
        "K5": "KIẾM PHONG KIM",
        "H6": "SƠN ÐẦU HỎA",
        "O6": "ỐC THƯỢNG THỔ",
        "M6": "BÌNH ÐỊA MỘC",
        "K6": "XOA XUYẾN KIM",
        "T6": "ÐẠI HẢI THỦY"}
    matranNapAm = [
        [0, "G", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "N", "Q"],
        [1, "K1", False, "T1", False, "H1", False, "O1", False, "M1", False],
        [2, False, "K1", False, "T1", False, "H1", False, "O1", False, "M1"],
        [3, "T2", False, "H2", False, "O2", False, "M2", False, "K2", False],
        [4, False, "T2", False, "H2", False, "O2", False, "M2", False, "K2"],
        [5, "H3", False, "O3", False, "M3", False, "K3", False, "T3", False],
        [6, False, "H3", False, "O3", False, "M3", False, "K3", False, "T3"],
        [7, "K4", False, "T4", False, "H4", False, "O4", False, "M4", False],
        [8, False, "K4", False, "T4", False, "H4", False, "O4", False, "M4"],
        [9, "T5", False, "H5", False, "O5", False, "M5", False, "K5", False],
        [10, False, "T5", False, "H5", False, "O5", False, "M5", False, "K5"],
        [11, "H6", False, "O6", False, "M6", False, "K6", False, "T6", False],
        [12, False, "H6", False, "O6", False, "M6", False, "K6", False, "T6"]
    ]
    try:
        nh = matranNapAm[diaChi][thienCan]
        if nh[0] in ["K", "M", "T", "H", "O"]:
            if xuatBanMenh is True:
                return banMenh[nh]
            else:
                return nh[0]
    except:
        raise Exception(nguHanhNapAm.__doc__)