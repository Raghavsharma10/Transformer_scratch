def AddMemberDefinition(self, member_definition):
    """Adds a member definition.

    Args:
      member_definition (DataTypeDefinition): member data type definition.
    """
    self.members.append(member_definition)
    member_definition.family_definition = self