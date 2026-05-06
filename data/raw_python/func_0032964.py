def add(self, schema, data):
        """ Stage ``data`` as a set of statements, based on the given
        ``schema`` definition. """
        binding = self.get_binding(schema, data)
        uri, triples = triplify(binding)
        for triple in triples:
            self.graph.add(triple)
        return uri