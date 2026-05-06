def get_name(self, data):
        """ For non-specific queries, this will return the actual name in the
        result. """
        if self.node.specific_attribute:
            return self.node.name
        name = data.get(self.predicate_var)
        if str(RDF.type) in [self.node.name, name]:
            return '$schema'
        if name.startswith(PRED):
            name = name[len(PRED):]
        return name