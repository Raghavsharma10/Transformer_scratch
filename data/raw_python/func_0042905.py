def tokenize_tags(tags_string):
    """
    This function is responsible to extract usable tags from a text.
    :param tags_string: a string of text
    :return: a string of comma separated tags
    """

    # text is parsed in two steps:
    # the first step extract every single world that is 3 > chars long
    # and that contains only alphanumeric characters, underscores and dashes
    tags_string = tags_string.lower().strip(",")
    single_words = set([w[:100] for w in re.split(';|,|\*|\n| ', tags_string)
                          if len(w) >= 3 and re.match("^[A-Za-z0-9_-]*$", w)])
    # the second step divide the original string using comma as separator
    comma_separated = set([t[:100] for t in tags_string.split(",") if t])
    # resulting set are merged using union
    return list(single_words | comma_separated)