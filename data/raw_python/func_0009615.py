def parse_vhdl_file(fname):
  '''Parse a named VHDL file
  
  Args:
    fname(str): Name of file to parse
  Returns:
    Parsed objects.
  '''
  with open(fname, 'rt') as fh:
    text = fh.read()
  return parse_vhdl(text)