def get_month_from_date_str(date_str, lang=DEFAULT_DATE_LANG):
    """Find the month name for the given locale, in the given string.

    Returns a tuple ``(number_of_month, abbr_name)``.
    """
    date_str = date_str.lower()
    with calendar.different_locale(LOCALES[lang]):
        month_abbrs = list(calendar.month_abbr)
        for seq, abbr in enumerate(month_abbrs):
            if abbr and abbr.lower() in date_str:
                return seq, abbr
    return ()