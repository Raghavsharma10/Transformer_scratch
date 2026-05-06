def get_required_fields(self):
    """Return the names of fields that are required according to the schema."""
    return [m.name for m in self._ast_node.members if m.member_schema.required]