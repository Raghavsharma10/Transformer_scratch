def int_to_gematria(num, gershayim=True):
    """convert integers between 1 an 999 to Hebrew numerals.

           - set gershayim flag to False to ommit gershayim
    """
    # 1. Lookup in specials
    if num in specialnumbers['specials']:
        retval = specialnumbers['specials'][num]
        return _add_gershayim(retval) if gershayim else retval

    # 2. Generate numeral normally
    parts = []
    rest = str(num)
    while rest:
        digit = int(rest[0])
        rest = rest[1:]
        if digit == 0:
            continue
        power = 10 ** len(rest)
        parts.append(specialnumbers['numerals'][power * digit])
    retval = ''.join(parts)
    # 3. Add gershayim
    return _add_gershayim(retval) if gershayim else retval