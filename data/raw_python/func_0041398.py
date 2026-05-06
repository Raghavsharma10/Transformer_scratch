def replace_month_abbr_with_num(date_str, lang=DEFAULT_DATE_LANG):
    """Replace month strings occurrences with month number."""
    num, abbr = get_month_from_date_str(date_str, lang)
    return re.sub(abbr, str(num), date_str, flags=re.IGNORECASE)