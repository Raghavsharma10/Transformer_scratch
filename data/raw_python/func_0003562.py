def camel_case_to_under_score(camel_case_name):
    """Return the underscore name of a camel case name.

    :param str camel_case_name: a name in camel case such as
      ``"ACamelCaseName"``
    :return: the name using underscores, e.g. ``"a_camel_case_name"``
    :rtype: str
    """
    result = []
    for letter in camel_case_name:
        if letter.lower() != letter:
            result.append("_" + letter.lower())
        else:
            result.append(letter.lower())
    if result[0].startswith("_"):
        result[0] = result[0][1:]
    return "".join(result)