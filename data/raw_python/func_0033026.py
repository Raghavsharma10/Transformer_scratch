def generate(self):
        """ Add provenance info to the context graph. """
        t = (self.context.identifier, RDF.type, META.Provenance)
        if t not in self.context.graph:
            self.context.graph.add(t)
        for name, value in self.data.items():
            pat = (self.context.identifier, META[name], None)
            if pat in self.context.graph:
                self.context.graph.remove(pat)
            self.context.graph.add((pat[0], META[name], Literal(value)))