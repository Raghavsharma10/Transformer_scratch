def _check_date_format(date, api):
    ''' Checks that the given date string conforms to the given API's date format specification '''
    try:
        datetime.datetime.strptime(date, api.DATE_FORMAT)
    except ValueError:
        raise ValueError("Date '{}' does not conform to API format: {}".format(date, api.DATE_FORMAT))