def _DecodeUrlSafe(urlsafe):
  """Decode a url-safe base64-encoded string.

  This returns the decoded string.
  """
  if not isinstance(urlsafe, basestring):
    raise TypeError('urlsafe must be a string; received %r' % urlsafe)
  if isinstance(urlsafe, unicode):
    urlsafe = urlsafe.encode('utf8')
  mod = len(urlsafe) % 4
  if mod:
    urlsafe += '=' * (4 - mod)
  # This is 3-4x faster than urlsafe_b64decode()
  return base64.b64decode(urlsafe.replace('-', '+').replace('_', '/'))