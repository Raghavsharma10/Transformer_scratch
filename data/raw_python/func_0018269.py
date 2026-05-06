def _GetTimeValues(self, number_of_seconds):
    """Determines time values.

    Args:
      number_of_seconds (int|decimal.Decimal): number of seconds.

    Returns:
       tuple[int, int, int, int]: days, hours, minutes, seconds.
    """
    number_of_seconds = int(number_of_seconds)
    number_of_minutes, seconds = divmod(number_of_seconds, 60)
    number_of_hours, minutes = divmod(number_of_minutes, 60)
    number_of_days, hours = divmod(number_of_hours, 24)
    return number_of_days, hours, minutes, seconds