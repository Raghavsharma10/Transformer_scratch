def search_weekday(weekday, jd, direction, offset):
    '''Determine the Julian date for the next or previous weekday'''
    return weekday_before(weekday, jd + (direction * offset))