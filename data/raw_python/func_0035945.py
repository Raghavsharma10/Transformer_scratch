def flip_uuid_parts(uuid):
  """
  Flips high and low segments of the timestamp portion of a UUID string.
  This enables correct lexicographic sorting. Because it is a simple flip,
  this function works in both directions.
  """
  flipped_uuid = uuid.split('-')
  flipped_uuid[0], flipped_uuid[2] = flipped_uuid[2], flipped_uuid[0]
  flipped_uuid = '-'.join(flipped_uuid)
  return flipped_uuid