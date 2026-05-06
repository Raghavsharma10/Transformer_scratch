def sorted_query_paths(self):
        """
        RETURN A LIST OF ALL SCHEMA'S IN DEPTH-FIRST TOPOLOGICAL ORDER
        """
        return list(reversed(sorted(p[0] for p in self.namespace.alias_to_query_paths.get(self.name))))