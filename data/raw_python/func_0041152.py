def add(TargetGroup, NewMember, Config=None, Args=None):
  r"""Adds members to an existing group.

  Args:
    TargetGroup (Group): The target group for the addition.
    NewMember (Group / Task): The member to be added.
    Config (dict): The config for the member.
    Args (OrderedDict): ArgConfig for the NewMember, if it's a task (optional).
  """
  Member = Task(NewMember, Args or {}, Config or {}) if isfunction(NewMember) else Group(NewMember, Config or {})
  ParentMembers = TargetGroup.__ec_member__.Members

  ParentMembers[Member.Config['name']] = Member

  alias = Member.Config.get('alias')

  if alias:
    ParentMembers[alias] = Member