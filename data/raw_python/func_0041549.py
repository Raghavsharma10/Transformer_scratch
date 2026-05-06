def member(Imported, **Config):
  r"""Helps with adding imported members to Scripts.

  Note:
    Config depends upon the Imported. It could be that of a **task** or a **group**.
  """
  __ec_member__ = Imported.__ec_member__
  __ec_member__.Config.update(**Config)

  state.ActiveModuleMemberQ.insert(0, __ec_member__)