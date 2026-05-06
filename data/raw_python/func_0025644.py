def weeks_per_year(year):
    '''Number of ISO weeks in a year'''
    # 53 weeks: any year starting on Thursday and any leap year starting on Wednesday
    jan1 = jwday(gregorian.to_jd(year, 1, 1))

    if jan1 == THU or (jan1 == WED and isleap(year)):
        return 53
    else:
        return 52