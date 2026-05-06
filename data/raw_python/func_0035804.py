def validate_stream(stream):
  """
  Check that the stream name is well-formed.
  """
  if not STREAM_REGEX.match(stream) or len(stream) > MAX_STREAM_LENGTH:
    raise InvalidStreamName(stream)