def thanksgiving(year, country='usa'):
    '''USA: last Thurs. of November, Canada: 2nd Mon. of October'''
    if country == 'usa':
        if year in [1940, 1941]:
            return nth_day_of_month(3, THU, NOV, year)
        elif year == 1939:
            return nth_day_of_month(4, THU, NOV, year)
        else:
            return nth_day_of_month(0, THU, NOV, year)

    if country == 'canada':
        return nth_day_of_month(2, MON, OCT, year)