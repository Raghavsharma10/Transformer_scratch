def se_clearing_code_bank_info(clearing: str) -> (str, int):
    """
    Returns Sweden bank info by clearning code.
    :param clearing: 4-digit clearing code
    :return: (Bank name, account digit count) or ('', None) if not found
    """
    from jutil.bank_const_se import SE_BANK_CLEARING_LIST
    for name, begin, end, acc_digits in SE_BANK_CLEARING_LIST:
        if begin <= clearing <= end:
            return name, acc_digits
    return '', None