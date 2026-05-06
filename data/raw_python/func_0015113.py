def get_name_from_abbrev(abbrev, case_sensitive=False):
    """
    Given a country code abbreviation, get the full name from the table.

    abbrev: (str) Country code to retrieve the full name of.
    case_sensitive: (bool) When True, enforce case sensitivity.
    """
    if case_sensitive:
        country_code = abbrev
    else:
        country_code = abbrev.upper()

    for code, full_name in COUNTRY_TUPLES:
        if country_code == code:
            return full_name

    raise KeyError('No country with that country code.')