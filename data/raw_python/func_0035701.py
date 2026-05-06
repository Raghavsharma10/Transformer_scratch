def cancel(task_id, secret_key=None, url=None):
  """Cancel scheduled task with `task_id`"""
  if not secret_key:
    secret_key = default_key()
  if not url:
    url = default_url()

  url = '%s/cancel' % url
  values = {
    'id': task_id,
  }
  return _send_with_auth(values, secret_key, url)