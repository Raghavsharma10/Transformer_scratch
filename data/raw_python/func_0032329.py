def get_binding(self, schema, data):
        """ For a given schema, get a binding mediator providing links to the
        RDF terms matching that schema. """
        schema = self.parent.get_schema(schema)
        return Binding(schema, self.parent.resolver, data=data)