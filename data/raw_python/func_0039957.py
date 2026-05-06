def get_memfree(memory, parallel):
  """Computes the memory required for the memfree field."""
  number = int(memory.rstrip(string.ascii_letters))
  memtype = memory.lstrip(string.digits)
  if not memtype:
    memtype = "G"
  return "%d%s" % (number*parallel, memtype)