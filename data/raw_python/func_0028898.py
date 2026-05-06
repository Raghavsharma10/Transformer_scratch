def logs(awsclient, function_name, start_dt, end_dt=None, tail=False):
    """Send a ping request to a lambda function.

    :param awsclient:
    :param function_name:
    :param start_dt:
    :param end_dt:
    :param tail:
    :return:
    """
    log.debug('Getting cloudwatch logs for: %s', function_name)
    log_group_name = '/aws/lambda/%s' % function_name

    current_date = None
    start_ts = datetime_to_timestamp(start_dt)
    if end_dt:
        end_ts = datetime_to_timestamp(end_dt)
    else:
        end_ts = None

    # tail mode
    # we assume that logs can arrive late but not out of order
    # so we hold the timestamp of the last logentry and start the next iteration
    # from there
    while True:
        logentries = filter_log_events(awsclient, log_group_name,
                                       start_ts=start_ts, end_ts=end_ts)
        if logentries:
            for e in logentries:
                actual_date, actual_time = decode_format_timestamp(e['timestamp'])
                if current_date != actual_date:
                    # print the date only when it changed
                    current_date = actual_date
                    log.info(current_date)
                log.info('%s  %s' % (actual_time, e['message'].strip()))
        if tail:
            if logentries:
                start_ts = logentries[-1]['timestamp'] + 1
            time.sleep(2)
            continue
        break