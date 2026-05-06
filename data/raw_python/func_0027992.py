def AddMemberDefinition(self, member_definition):
    """Adds a member definition.

    Args:
      member_definition (DataTypeDefinition): member data type definition.
    """
    self._byte_size = None
    self.members.append(member_definition)

    if self.sections:
      section_definition = self.sections[-1]
      section_definition.members.append(member_definition)