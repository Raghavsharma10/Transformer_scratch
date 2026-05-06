def print_invalid_chars(invalid_chars, vargs):
    """
    Print Unicode characterss that are not IPA valid,
    if requested by the user.

    :param list invalid_chars: a list (possibly empty) of invalid Unicode characters
    :param dict vargs: the command line parameters
    """
    if len(invalid_chars) > 0:
        if vargs["print_invalid"]:
            print(u"".join(invalid_chars))
        if vargs["unicode"]:
            for u_char in sorted(set(invalid_chars)):
                print(u"'%s'\t%s\t%s" % (u_char, hex(ord(u_char)), unicodedata.name(u_char, "UNKNOWN")))