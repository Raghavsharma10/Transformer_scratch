def get_schema_spec(self, key):
    """Return the evaluated schema expression from a subkey."""
    member_node = self._ast_node.member.get(key, None)
    if not member_node:
      return schema.AnySchema()

    s = framework.eval(member_node.member_schema, self.env(self))
    if not isinstance(s, schema.Schema):
      raise ValueError('Node %r with schema node %r should evaluate to Schema, got %r' % (member_node, member_node.member_schema, s))
    return s