def iban_bank_info(v: str) -> (str, str):
    """
    Returns BIC code and bank name from IBAN number.
    :param v: IBAN account number
    :return: (BIC code, bank name) or ('', '') if not found / unsupported country
    """
    v = iban_filter(v)
    if v[:2] == 'FI':
        return fi_iban_bank_info(v)
    else:
        return '', ''