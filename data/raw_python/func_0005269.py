def fi_payment_reference_number(num: str):
    """
    Appends Finland reference number checksum to existing number.
    :param num: At least 3 digits
    :return: Number plus checksum
    """
    assert isinstance(num, str)
    num = STRIP_WHITESPACE.sub('', num)
    num = re.sub(r'^0+', '', num)
    assert len(num) >= 3
    weights = [7, 3, 1]
    weighed_sum = 0
    numlen = len(num)
    for j in range(numlen):
        weighed_sum += int(num[numlen - 1 - j]) * weights[j % 3]
    return num + str((10 - (weighed_sum % 10)) % 10)