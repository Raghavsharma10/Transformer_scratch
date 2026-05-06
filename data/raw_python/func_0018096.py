def attach_schema(self, schem):
    """Add a tuple schema to this object (externally imposed)"""
    self.tuple_schema = schema.AndSchema.make(self.tuple_schema, schem)