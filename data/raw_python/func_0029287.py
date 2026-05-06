def prettify(string):
    """
    Turns an ugly text string into a beautiful one by applying a regex pipeline which ensures the following:

    - String cannot start or end with spaces
    - String cannot have multiple sequential spaces, empty lines or punctuation (except for "?", "!" and ".")
    - Arithmetic operators (+, -, /, \*, =) must have one, and only one space before and after themselves
    - The first letter after a dot, an exclamation or a question mark must be uppercase
    - One, and only one space should follow a dot, an exclamation or a question mark
    - Text inside double quotes cannot start or end with spaces, but one, and only one space must come first and \
    after quotes (foo" bar"baz -> foo "bar" baz)
    - Text inside round brackets cannot start or end with spaces, but one, and only one space must come first and \
    after brackets ("foo(bar )baz" -> "foo (bar) baz")
    - Percentage sign ("%") cannot be preceded by a space if there is a number before ("100 %" -> "100%")
    - Saxon genitive is correct ("Dave' s dog" -> "Dave's dog")


    :param string: String to manipulate
    :return: Prettified string.
    :rtype: str
    """

    def remove_duplicates(regex_match):
        return regex_match.group(1)[0]

    def uppercase_first_letter_after_sign(regex_match):
        match = regex_match.group(1)
        return match[:-1] + match[2].upper()

    def ensure_right_space_only(regex_match):
        return regex_match.group(1).strip() + ' '

    def ensure_left_space_only(regex_match):
        return ' ' + regex_match.group(1).strip()

    def ensure_spaces_around(regex_match):
        return ' ' + regex_match.group(1).strip() + ' '

    def remove_internal_spaces(regex_match):
        return regex_match.group(1).strip()

    def fix_saxon_genitive(regex_match):
        return regex_match.group(1).replace(' ', '') + ' '

    p = PRETTIFY_RE['DUPLICATES'].sub(remove_duplicates, string)
    p = PRETTIFY_RE['RIGHT_SPACE'].sub(ensure_right_space_only, p)
    p = PRETTIFY_RE['LEFT_SPACE'].sub(ensure_left_space_only, p)
    p = PRETTIFY_RE['SPACES_AROUND'].sub(ensure_spaces_around, p)
    p = PRETTIFY_RE['SPACES_INSIDE'].sub(remove_internal_spaces, p)
    p = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].sub(uppercase_first_letter_after_sign, p)
    p = PRETTIFY_RE['SAXON_GENITIVE'].sub(fix_saxon_genitive, p)
    p = p.strip()
    try:
        return p[0].capitalize() + p[1:]
    except IndexError:
        return p