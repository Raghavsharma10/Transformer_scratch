def get(self, id, depth=3, schema=None):
        """ Construct a single object based on its ID. """
        uri = URIRef(id)
        if schema is None:
            for o in self.graph.objects(subject=uri, predicate=RDF.type):
                schema = self.parent.get_schema(str(o))
                if schema is not None:
                    break
        else:
            schema = self.parent.get_schema(schema)
        binding = self.get_binding(schema, None)
        return self._objectify(uri, binding, depth=depth, path=set())