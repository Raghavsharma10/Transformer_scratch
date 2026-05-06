def fi_iban_bank_info(v: str) -> (str, str):
    """
    Returns BIC code and bank name from FI IBAN number.
    :param v: IBAN account number
    :return: (BIC code, bank name) or ('', '') if not found
    """
    from jutil.bank_const_fi import FI_BIC_BY_ACCOUNT_NUMBER, FI_BANK_NAME_BY_BIC
    v = iban_filter(v)
    bic = FI_BIC_BY_ACCOUNT_NUMBER.get(v[4:7], None)
    return (bic, FI_BANK_NAME_BY_BIC[bic]) if bic is not None else ('', '')