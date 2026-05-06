def slugify(string, sign='-'):
    """
    Converts a string into a slug using provided join sign.
    (**(This Is A "Test"!)** -> **this-is-a-test**)

    :param string: String to convert.
    :type string: str
    :param sign: Sign used to join string tokens (default to "-").
    :type sign: str
    :return: Slugified string
    """
    if not is_string(string):
        raise TypeError('Expected string')

    # unicode casting for python 2 (unicode is default for python 3)
    try:
        string = unicode(string, 'utf-8')
    except NameError:
        pass

    # replace any character that is NOT letter or number with spaces
    s = NO_LETTERS_OR_NUMBERS_RE.sub(' ', string.lower()).strip()

    # replace spaces with join sign
    s = SPACES_RE.sub(sign, s)

    # normalize joins (remove duplicates)
    s = re.sub(re.escape(sign) + r'+', sign, s)

    # translate non-ascii signs
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8')

    return s