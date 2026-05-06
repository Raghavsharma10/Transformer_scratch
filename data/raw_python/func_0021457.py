def nguHanh(tenHanh):
    """
    Args:
        tenHanh (string): Tên Hành trong ngũ hành, Kim hoặc K, Moc hoặc M,
        Thuy hoặc T, Hoa hoặc H, Tho hoặc O

    Returns:
        Dictionary: ID của Hành, tên đầy đủ của Hành, số Cục của Hành

    Raises:
        Exception: Description
    """
    if tenHanh in ["Kim", "K"]:
        return {"id": 1, "tenHanh": "Kim", "cuc": 4, "tenCuc": "Kim tứ Cục",
                "css": "hanhKim"}
    elif tenHanh == "Moc" or tenHanh == "M":
        return {"id": 2, "tenHanh": "Mộc", "cuc": 3, "tenCuc": "Mộc tam Cục",
                "css": "hanhMoc"}
    elif tenHanh == "Thuy" or tenHanh == "T":
        return {"id": 3, "tenHanh": "Thủy", "cuc": 2, "tenCuc": "Thủy nhị Cục",
                "css": "hanhThuy"}
    elif tenHanh == "Hoa" or tenHanh == "H":
        return {"id": 4, "tenHanh": "Hỏa", "cuc": 6, "tenCuc": "Hỏa lục Cục",
                "css": "hanhHoa"}
    elif tenHanh == "Tho" or tenHanh == "O":
        return {"id": 5, "tenHanh": "Thổ", "cuc": 5, "tenCuc": "Thổ ngũ Cục",
                "css": "hanhTho"}
    else:
        raise Exception(
            "Tên Hành phải thuộc Kim (K), Mộc (M), Thủy (T), \
             Hỏa (H) hoặc Thổ (O)")