def parse_verilog_file(fname):
  '''Parse a named Verilog file
  
  Args:
    fname (str): File to parse.
  Returns:
    List of parsed objects.
  '''
  with open(fname, 'rt') as fh:
    text = fh.read()
  return parse_verilog(text)