def setActiveModule(Module):
  r"""Helps with collecting the members of the imported modules.
  """
  module_name = Module.__name__

  if module_name not in ModuleMembers:
    ModuleMembers[module_name] = []
    ModulesQ.append(module_name)
    Group(Module, {}) # brand the module with __ec_member__

  state.ActiveModuleMemberQ = ModuleMembers[module_name]