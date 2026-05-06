def is_valid_country_abbrev(abbrev, case_sensitive=False):
    """
    Given a country code abbreviation, check to see if it matches the
    country table.

    abbrev: (str) Country code to evaluate.
    case_sensitive: (bool) When True, enforce case sensitivity.

    Returns True if valid, False if not.
    """
    if case_sensitive:
        country_code = abbrev
    else:
        country_code = abbrev.upper()

    for code, full_name in COUNTRY_TUPLES:
        if country_code == code:
            return True

    return False