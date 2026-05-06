def format_timedelta(dt: timedelta) -> str:
    """
    Formats timedelta to readable format, e.g. 1h30min.
    :param dt: timedelta
    :return: str
    """
    seconds = int(dt.total_seconds())
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    s = ""
    if days > 0:
        s += str(days) + "d"
    if hours > 0:
        s += str(hours) + "h"
    if minutes > 0:
        s += str(minutes) + "min"
    if s == "":
        s = "0min"
    return s