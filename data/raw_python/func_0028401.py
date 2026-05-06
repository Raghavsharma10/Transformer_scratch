def get_readable_time_string(seconds):
    """Returns human readable string from number of seconds"""
    seconds = int(seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    days = hours // 24
    hours = hours % 24

    result = ""
    if days > 0:
        result += "%d %s " % (days, "Day" if (days == 1) else "Days")
    if hours > 0:
        result += "%d %s " % (hours, "Hour" if (hours == 1) else "Hours")
    if minutes > 0:
        result += "%d %s " % (minutes, "Minute" if (minutes == 1) else "Minutes")
    if seconds > 0:
        result += "%d %s " % (seconds, "Second" if (seconds == 1) else "Seconds")

    return result.strip()