def _get_timeframe_bounds(self, timeframe, bucket_width):
    """
    Get a `bucket_width` aligned `start_time` and `end_time` from a
    `timeframe` dict
    """
    if bucket_width:
      bucket_width_seconds = bucket_width
      bucket_width = epoch_time_to_kronos_time(bucket_width)

    # TODO(derek): Potential optimization by setting the end_time equal to the
    # untrusted_time if end_time > untrusted_time and the results are not being
    # output to the user (only for caching)
    if timeframe['mode']['value'] == 'recent':
      # Set end_time equal to now and align to bucket width
      end_time = kronos_time_now()
      original_end_time = end_time
      duration = get_seconds(timeframe['value'], timeframe['scale']['name'])
      duration = epoch_time_to_kronos_time(duration)
      start_time = original_end_time - duration

      if bucket_width:
        # Align values to the bucket width
        # TODO(derek): Warn the user that the timeframe has been altered to fit
        # the bucket width
        if (end_time % bucket_width) != 0:
          end_time += bucket_width - (end_time % bucket_width)

        if (start_time % bucket_width) != 0:
          start_time -= (start_time % bucket_width)

      start = kronos_time_to_datetime(start_time)
      end = kronos_time_to_datetime(end_time)
    elif timeframe['mode']['value'] == 'range':
      end = datetime.datetime.strptime(timeframe['to'], DT_FORMAT)
      end_seconds = datetime_to_epoch_time(end)

      start = datetime.datetime.strptime(timeframe['from'], DT_FORMAT)
      start_seconds = datetime_to_epoch_time(start)

      if bucket_width:
        # Align values to the bucket width
        # TODO(derek): Warn the user that the timeframe has been altered to fit
        # the bucket width
        start_bump = start_seconds % bucket_width_seconds
        start -= datetime.timedelta(seconds=start_bump)
        if (end_seconds % bucket_width_seconds) != 0:
          end_bump = bucket_width_seconds - (end_seconds % bucket_width_seconds)
          end += datetime.timedelta(seconds=end_bump)
    else:
      raise ValueError("Timeframe mode must be 'recent' or 'range'")

    return start, end