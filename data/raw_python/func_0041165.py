def exclude_chars(text, exclusion=None):
    """
    Clean text string of simbols in exclusion list.
    """
    exclusion = [] if exclusion is None else exclusion
    regexp = r"|".join([select_regexp_char(x) for x in exclusion]) or r''
    return re.sub(regexp, '', text)