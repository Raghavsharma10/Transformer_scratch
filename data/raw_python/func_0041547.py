def group(__decorated__, **Config):
  r"""A decorator to make groups out of classes.

  Config:
    * name (str): The name of the group. Defaults to __decorated__.__name__.
    * desc (str): The description of the group (optional).
    * alias (str): The alias for the group (optional).
  """
  _Group = Group(__decorated__, Config)

  if isclass(__decorated__): # convert the method of the class to static methods so that they could be accessed like object methods; ir: g1/t1(...).
    static(__decorated__)

  state.ActiveModuleMemberQ.insert(0, _Group)

  return _Group.Underlying