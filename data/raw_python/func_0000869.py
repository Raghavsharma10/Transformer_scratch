def escape_ampersand(string):
    """
    Quick convert unicode ampersand characters not associated with
    a numbered entity or not starting with allowed characters to a plain &amp;
    """
    if not string:
        return string
    start_with_match = r"(\#x(....);|lt;|gt;|amp;)"
    # The pattern below is match & that is not immediately followed by #
    string = re.sub(r"&(?!" + start_with_match + ")", '&amp;', string)
    return string