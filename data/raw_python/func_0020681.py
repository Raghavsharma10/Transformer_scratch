def string_to_level(log_level):
  """
  Converts a string to the corresponding log level
  """
  if (log_level.strip().upper() == "DEBUG"):
    return logging.DEBUG
  if (log_level.strip().upper() == "INFO"):
    return logging.INFO
  if (log_level.strip().upper() == "WARNING"):
    return logging.WARNING
  if (log_level.strip().upper() == "ERROR"):
    return logging.ERROR