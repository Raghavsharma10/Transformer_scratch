def str_on_2_unicode_on_3(s):
    """
    argparse is way too awesome when doing repr() on choices when printing usage

    :param s: str or unicode
    :return: str on 2, unicode on 3
    """

    if not PY3:
        return str(s)
    else:  # 3+
        if not isinstance(s, str):
            return str(s, encoding="utf-8")
        return s