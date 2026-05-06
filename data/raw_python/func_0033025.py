def _load(self):
        """ Load provenance info from the main store. """
        graph = self.context.parent.graph.get_context(self.context.identifier)
        data = {}
        for (_, p, o) in graph.triples((self.context.identifier, None, None)):
            if not p.startswith(META):
                continue
            name = p[len(META):]
            data[name] = o.toPython()
        return data