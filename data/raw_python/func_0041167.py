def normalizer(text, exclusion=OPERATIONS_EXCLUSION, lower=True, separate_char='-', **kwargs):
    """
    Clean text string of simbols only alphanumeric chars.
    """
    clean_str = re.sub(r'[^\w{}]'.format(
        "".join(exclusion)), separate_char, text.strip()) or ''
    clean_lowerbar = clean_str_without_accents = strip_accents(clean_str)

    if '_' not in exclusion:
        clean_lowerbar = re.sub(r'\_', separate_char, clean_str_without_accents.strip())

    limit_guion = re.sub(r'\-+', separate_char, clean_lowerbar.strip())

    # TODO: refactor with a regexp
    if limit_guion and separate_char and separate_char in limit_guion[0]:
        limit_guion = limit_guion[1:]

    if limit_guion and separate_char and separate_char in limit_guion[-1]:
        limit_guion = limit_guion[:-1]

    if lower:
        limit_guion = limit_guion.lower()

    return limit_guion