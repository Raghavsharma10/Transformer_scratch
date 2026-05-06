def _iter_candidate_groups(self, init_match, edges0, edges1):
        """Divide the edges into groups"""
        # collect all end vertices0 and end vertices1 that belong to the same
        # group.
        sources = {}
        for start_vertex0, end_vertex0 in edges0:
            l = sources.setdefault(start_vertex0, [])
            l.append(end_vertex0)
        dests = {}
        for start_vertex1, end_vertex1 in edges1:
            start_vertex0 = init_match.reverse[start_vertex1]
            l = dests.setdefault(start_vertex0, [])
            l.append(end_vertex1)
        for start_vertex0, end_vertices0 in sources.items():
            end_vertices1 = dests.get(start_vertex0, [])
            yield end_vertices0, end_vertices1