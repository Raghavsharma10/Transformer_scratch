def listMemberHelps(TargetGroup):
  r"""Gets help on a group's children.
  """
  Members = []

  for Member in TargetGroup.Members.values(): # get unique children (by discarding aliases)
    if Member not in Members:
      Members.append(Member)

  Ret = []

  for Member in Members:
    Config = Member.Config
    Ret.append(('%s%s' % (Config['name'], ', %s' % Config['alias'] if 'alias' in Config else ''), Config.get('desc', '')))

  return Ret