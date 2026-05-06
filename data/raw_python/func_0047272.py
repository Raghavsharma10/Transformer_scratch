def apply_constraints(phash, size, nonalphanumeric):
    """
    Fiddle with the password a bit after hashing it so that it will
    get through most website filters. We require one upper and lower
    case, one digit, and we look at the user's password to determine
    if there should be at least one alphanumeric or not.
    """
    starting_size = max(0, size - 4)
    result = phash[:starting_size]

    extras = itertools.chain((ord(ch) for ch in phash[starting_size:]),
                             itertools.repeat(0))
    extra_chars = (chr(ch) for ch in extras)
    nonword = re.compile(r'\W')

    def next_between(start, end):
        interval = ord(end) - ord(start) + 1
        offset = next(extras) % interval
        return chr(ord(start) + offset)

    for elt, repl in (
        (re.compile('[A-Z]'), lambda: next_between('A', 'Z')),
        (re.compile('[a-z]'), lambda: next_between('a', 'z')),
        (re.compile('[0-9]'), lambda: next_between('0', '9'))):
        if len(elt.findall(result)) != 0:
            result += next(extra_chars)
        else:
            result += repl()

    if len(nonword.findall(result)) != 0 and nonalphanumeric:
        result += next(extra_chars)
    else:
        result += '+'

    while len(nonword.findall(result)) != 0 and not nonalphanumeric:
        result = nonword.sub(next_between('A', 'Z'), result, 1)

    amount = next(extras) % len(result)
    result = result[amount:] + result[0:amount]

    return result