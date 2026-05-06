def get_numbers(s):
    """Extracts all integers from a string an return them in a list"""

    result = map(int, re.findall(r'[0-9]+', unicode(s)))
    return result + [1] * (2 - len(result))