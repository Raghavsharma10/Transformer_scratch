def mkApplications(location, *atoms):
  """Make a sequence of applications from a list of tokens.

  atoms is a list of atoms, which will be handled left-associatively. E.g:

      ['foo', [], []] == foo()() ==> Application(Application('foo', []), [])
  """
  atoms = list(atoms)
  while len(atoms) > 1:
    atoms[0:2] = [Application(location, atoms[0], atoms[1])]

  # Nothing left to apply
  return atoms[0]