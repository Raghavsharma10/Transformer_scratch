def make_machine_mapping(machine_list):
  """
  Convert the machine list argument from a list of names into a mapping of logical names to
  physical hosts. This is similar to the _parse_configs function but separated to provide the
  opportunity for extension and additional checking of machine access
  """
  if machine_list is None:
    return {}
  else:
    mapping = {}
    for pair in machine_list:
      if (constants.MACHINE_SEPARATOR not in pair) or (pair.count(constants.MACHINE_SEPARATOR) != 1):
        raise ValueError("machine pairs must be passed as two strings separted by a %s", constants.MACHINE_SEPARATOR)
      (logical, physical) = pair.split(constants.MACHINE_SEPARATOR)
      # add checks for reachability
      mapping[logical] = physical
    return mapping