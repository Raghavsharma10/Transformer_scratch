def create_next_tag():
    """ creates a tag based on the date and previous tags """
    date = datetime.utcnow()
    date_tag = '{}.{}.{}'.format(date.year, date.month, date.day)
    if date_tag in latest_tag(): # if there was an update already today
        latest = latest_tag().split('.') # split by spaces
        if len(latest) == 4: # if it is not the first revision of the day
            latest[-1]= str(int(latest[-1])+1)
        else: # if it is the first revision of the day
            latest+=['1']
        date_tag = '.'.join(latest)
    return date_tag