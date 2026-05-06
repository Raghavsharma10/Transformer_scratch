def format_entry(record, show_level=False, colorize=False):
    """
    Format a log entry according to its level and context
    """
    if show_level:
        log_str = u'{}: {}'.format(record.levelname, record.getMessage())
    else:
        log_str = record.getMessage()

    if colorize and record.levelname in LOG_COLORS:
        log_str = u'<span color="{}">'.format(LOG_COLORS[record.levelname]) + log_str + u'</span>'

    return log_str