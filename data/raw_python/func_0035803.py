def validate_event_and_assign_id(event):
  """
  Ensure that the event has a valid time. Assign a random UUID based on the
  event time.
  """
  event_time = event.get(TIMESTAMP_FIELD)

  if event_time is None:
    event[TIMESTAMP_FIELD] = event_time = epoch_time_to_kronos_time(time.time())
  elif type(event_time) not in (int, long):
    raise InvalidEventTime(event_time)

  # Generate a uuid1-like sequence from the event time with the non-time bytes
  # set to random values.
  _id = uuid_from_kronos_time(event_time)
  event[ID_FIELD] = str(_id)
  return _id, event