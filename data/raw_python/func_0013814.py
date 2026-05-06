def do_set(parser, token):
  '''Calls an arbitrary method on an object.'''
  code = token.contents
  firstspace = code.find(' ')
  if firstspace >= 0:
    code = code[firstspace+1:]
  return Setter(code)