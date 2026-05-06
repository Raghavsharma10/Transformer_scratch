def use_backup_if_fail(app, key):
    """
    Return a error flag for prompt message in front-end  if failure times (unceasing fail)
    greater than max failure times else return backup data (latest data in the cache)
    """
    lock.acquire()
    try:
        if key not in backup:
            backup[key] = {}
        if key in fail_times and fail_times[key] % app.config[MAX_FAILURE_TIMES] == 0:
            logger.error(
                '<SERVER KEY %s> At present already reaching the upper limit of the max failure times, failure times: %s' % (
                    key, fail_times[key]))
            message = {app.config[MAX_FAILURE_MESSAGE_KEY]: MAX_FAILURE_MESSAGE % key}
            if alarm_email is not None:
                _send_alarm_email('Happened fault!', MAX_FAILURE_MESSAGE % key)
            return unite_dict(backup[key], message)
        else:
            logger.info('<SERVER KEY %s> Request fail or in a status of sleep time window and return backup data %s' % (
                key, backup[key]))
            return backup[key]
    finally:
        lock.release()