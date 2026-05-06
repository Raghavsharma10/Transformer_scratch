def jstimestamp(dte):
    '''Convert a date or datetime object into a javsacript timestamp.'''
    days = date(dte.year, dte.month, 1).toordinal() - _EPOCH_ORD + dte.day - 1
    hours = days*24
    
    if isinstance(dte,datetime):
        hours += dte.hour
        minutes = hours*60 + dte.minute
        seconds = minutes*60 + dte.second
        return 1000*seconds + int(0.001*dte.microsecond)
    else:
        return 3600000*hours