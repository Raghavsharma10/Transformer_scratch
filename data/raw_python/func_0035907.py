def enable_precompute(panel):
  """Schedule a precompute task for `panel`"""
  use_metis = panel['data_source']['source_type'] == 'querybuilder'
  if use_metis:
    query = panel['data_source']['query']
  else:
    query = "u'''%s'''" % panel['data_source']['code']
  precompute = panel['data_source']['precompute']
  timeframe = panel['data_source']['timeframe']
  bucket_width = precompute['bucket_width']['value']
  time_scale = precompute['bucket_width']['scale']['name']
  bucket_width_seconds = get_seconds(bucket_width, time_scale)

  if timeframe['mode']['value'] == 'recent':
    untrusted_time = precompute['untrusted_time']['value']
    untrusted_time_scale = precompute['untrusted_time']['scale']['name']
    untrusted_time_seconds = get_seconds(untrusted_time, untrusted_time_scale)
    # Schedule the task with an interval equal to the bucket_width
    interval = bucket_width_seconds
  elif timeframe['mode']['value'] == 'range':
    untrusted_time_seconds = 0
    # Schedule the task with an interval of 0 so it only runs once
    interval = 0

  task_code = PRECOMPUTE_INITIALIZATION_CODE % (query, timeframe,
                                                bucket_width_seconds,
                                                untrusted_time_seconds,
                                                use_metis)
  result = scheduler_client.schedule(task_code, interval)

  if result['status'] != 'success':
    raise RuntimeError(result.get('reason'))

  return result['id']