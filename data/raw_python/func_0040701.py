def resolve_module(module, definitions):
  """Resolve (through indirections) the program contents of a module definition.
  The result is a list of program chunks."""

  assert module in definitions, "No definition for module '%s'" % module
  
  d = definitions[module]
  if type(d) == dict:
    if 'filename' in d:
      with open(d['filename']) as f:
        return [f.read().strip()]
    elif 'reference' in d:
      return resolve_module(d['reference'], definitions)
    elif 'group' in d:
      return sum([resolve_module(m,definitions) for m in d['group']],[])
    else:
      assert False
  else:
    assert type(d) == str
    return [d]