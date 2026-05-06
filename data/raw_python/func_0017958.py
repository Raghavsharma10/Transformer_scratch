def fingerprint(value):
  """Return a hash value that uniquely identifies the GCL value."""
  h = hashlib.sha256()
  _digest(value, h)
  return h.digest().encode('hex')