def asdate(self):
    """Return this datetime_tz as a date object.

    Returns:
      This datetime_tz as a date object.
    """
    return datetime.date(self.year, self.month, self.day)