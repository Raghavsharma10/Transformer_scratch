def create_date(past=False, max_years_future=10, max_years_past=10):
    """
    Create a random valid date
    If past, then dates can be in the past
    If into the future, then no more than max_years into the future
    If it's not, then it can't be any older than max_years_past
    """
    if past:
        start = datetime.datetime.today() - datetime.timedelta(days=max_years_past * 365)
        #Anywhere between 1980 and today plus max_ears
        num_days = (max_years_future * 365) + start.day
    else:
        start = datetime.datetime.today()
        num_days = max_years_future * 365

    random_days = random.randint(1, num_days)
    random_date = start + datetime.timedelta(days=random_days)
    return(random_date)