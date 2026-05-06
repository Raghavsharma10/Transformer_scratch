def get_progress(start, finish):
    """
    Args:
        start (DateTime): start date
        finish (DateTime): finish date
    Returns:

    """
    now = datetime.now()
    dif_time_start = start - now
    dif_time_finish = finish - now

    if dif_time_start.days < 0 and dif_time_finish.days < 0:
        return PROGRESS_STATES[3][0]
    elif dif_time_start.days < 0 and dif_time_finish.days >= 1:
        return PROGRESS_STATES[2][0]
    elif dif_time_start.days >= 1 and dif_time_finish.days >= 1:
        return PROGRESS_STATES[0][0]
    else:
        return PROGRESS_STATES[2][0]