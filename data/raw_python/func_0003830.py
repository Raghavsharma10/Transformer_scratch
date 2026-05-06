def _get_name(self, graph, group=None):
        """Convert a molecular graph into a unique name

           This method is not sensitive to the order of the atoms in the graph.
        """
        if group is not None:
            graph = graph.get_subgraph(group, normalize=True)

        fingerprint = graph.fingerprint.tobytes()
        name = self.name_cache.get(fingerprint)
        if name is None:
            name = "NM%02i" % len(self.name_cache)
            self.name_cache[fingerprint] = name
        return name