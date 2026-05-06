def tzolkin_generator(number=None, name=None):
    '''For a given tzolkin name/number combination, return a generator
    that gives cycle, starting with the input'''

    # By default, it will start at the beginning
    number = number or 13
    name = name or "Ajaw"

    if number > 13:
        raise ValueError("Invalid day number")

    if name not in TZOLKIN_NAMES:
        raise ValueError("Invalid day name")

    count = _tzolkin_count(number, name)

    ranged = itertools.chain(list(range(count, 260)), list(range(1, count)))

    for i in ranged:
        yield _tzolkin_from_count(i)