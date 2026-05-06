def now(format_string):
    """
    Displays the date, formatted according to the given string.

    Uses the same format as PHP's ``date()`` function; see http://php.net/date
    for all the possible values.

    Sample usage::

        It is {% now "jS F Y H:i" %}
    """
    from datetime import datetime
    from django.utils.dateformat import DateFormat
    return DateFormat(datetime.now()).format(self.format_string)