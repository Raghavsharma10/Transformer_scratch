def GetDate(text=None, selected=None, **kwargs):
    """Prompt the user for a date.
    
    This will raise a Zenity Calendar Dialog for the user to pick a date.
    It will return a datetime.date object with the date or None if the 
    user hit cancel.
    
    text - Text to be displayed in the calendar dialog.
    selected - A datetime.date object that will be the pre-selected date.
    kwargs - Optional command line parameters for Zenity such as height,
             width, etc."""

    args = ['--date-format=%d/%m/%Y']
    if text:
        args.append('--text=%s' % text)
    if selected:
        args.append('--day=%d' % selected.day)
        args.append('--month=%d' % selected.month)
        args.append('--year=%d' % selected.year)

    for generic_args in kwargs_helper(kwargs):
        args.append('--%s=%s' % generic_args)

    p = run_zenity('--calendar', *args)

    if p.wait() == 0:
        retval = p.stdout.read().strip()
        day, month, year = [int(x) for x in retval.split('/')]
        return date(year, month, day)