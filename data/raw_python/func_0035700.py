def schedule(code, interval, secret_key=None, url=None):
  """Schedule a string of `code` to be executed every `interval`

  Specificying an `interval` of 0 indicates the event should only be run
  one time and will not be rescheduled.
  """
  if not secret_key:
    secret_key = default_key()
  if not url:
    url = default_url()

  url = '%s/schedule' % url
  values = {
    'interval': interval,
    'code': code,
  }
  return _send_with_auth(values, secret_key, url)