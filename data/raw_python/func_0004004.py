def command_chars(string, vargs):
    """
    Print a list of all IPA characters in the given string.

    It will print the Unicode representation, the full IPA name,
    and the Unicode "U+"-prefixed hexadecimal codepoint representation
    of each IPA character.

    :param str string: the string to act upon
    :param dict vargs: the command line arguments
    """
    try:
        ipa_string = IPAString(
            unicode_string=string,
            ignore=vargs["ignore"],
            single_char_parsing=vargs["single_char_parsing"]
        )
        for c in ipa_string:
            print(u"'%s'\t%s (%s)" % (c.unicode_repr, c.name, unicode_to_hex(c.unicode_repr)))
    except ValueError as exc:
        print_error(str(exc))