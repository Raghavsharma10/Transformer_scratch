def columbus_day(year, country='usa'):
    '''in USA: 2nd Monday in Oct
       Elsewhere: Oct 12'''
    if country == 'usa':
        return nth_day_of_month(2, MON, OCT, year)
    else:
        return (year, OCT, 12)