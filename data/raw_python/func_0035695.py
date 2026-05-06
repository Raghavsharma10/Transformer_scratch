def uuid_from_kronos_time(time, _type=UUIDType.RANDOM):
  """
  Generate a UUID with the specified time.
  If `lowest` is true, return the lexicographically first UUID for the specified
  time.
  """
  return timeuuid_from_time(int(time) + UUID_TIME_OFFSET, type=_type)