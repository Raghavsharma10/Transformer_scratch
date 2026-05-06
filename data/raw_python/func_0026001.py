def legal_date(year, month, day):
    '''Checks if a given date is a legal positivist date'''
    try:
        assert year >= 1
        assert 0 < month <= 14
        assert 0 < day <= 28
        if month == 14:
            if isleap(year + YEAR_EPOCH - 1):
                assert day <= 2
            else:
                assert day == 1

    except AssertionError:
        raise ValueError("Invalid Positivist date: ({}, {}, {})".format(year, month, day))

    return True