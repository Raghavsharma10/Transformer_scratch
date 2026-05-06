def delay_2(year):
    '''Check for delay in start of new year due to length of adjacent years'''

    last = delay_1(year - 1)
    present = delay_1(year)
    next_ = delay_1(year + 1)

    if next_ - present == 356:
        return 2
    elif present - last == 382:
        return 1
    else:
        return 0