def _iter_new_relations(self, init_match, subject_graph, edges0, constraints0, edges1):
        """Given an onset for a match, iterate over all possible new key-value pairs"""
        # Count the number of unique edges0[i][1] values. This is also
        # the number of new relations.
        num_new_relations = len(set(j for i, j in edges0))

        def combine_small(relations, num):
            """iterate over all compatible combinations within one set of relations"""
            if len(relations) == 0:
                return
            for i, pivot in enumerate(relations):
                if num == 1:
                    yield (pivot, )
                else:
                    compatible_relations = list(
                        item for item in relations[:i]
                        if pivot[0]!=item[0] and pivot[1]!=item[1]
                    )
                    for tail in combine_small(compatible_relations, num-1):
                        yield (pivot, ) + tail

        # generate candidate relations
        candidate_relations = []
        icg = self._iter_candidate_groups(init_match, edges0, edges1)
        for end_vertices0, end_vertices1 in icg:
            if len(end_vertices0) > len(end_vertices1):
                return # this can never work, the subject graph is 'too small'
            elif not self.pattern.sub and \
                 len(end_vertices0) != len(end_vertices1):
                return # an exact match is sought, this can never work
            l = []
            for end_vertex0 in end_vertices0:
                for end_vertex1 in end_vertices1:
                    if self.pattern.compare(end_vertex0, end_vertex1, subject_graph):
                        l.append((end_vertex0, end_vertex1))
            # len(end_vertices0) = the total number of relations that must be
            # made in this group
            if len(l) > 0:
                # turn l into a list of sets of internally compatible candidate
                # relations in this group
                l = list(combine_small(l, len(end_vertices0)))
                candidate_relations.append(l)
        if len(candidate_relations) == 0:
            return
        self.print_debug("candidate_relations: %s" % candidate_relations)

        def combine_big(pos=0):
            """Iterate over all possible sets of relations"""
            # pos is an index in candidate_relations
            crs = candidate_relations[pos]
            if pos == len(candidate_relations)-1:
                for relations in crs:
                    yield relations
            else:
                for tail in combine_big(pos+1):
                    for relations in crs:
                        yield relations + tail

        # final loop
        for new_relations in combine_big():
            new_relations = set(new_relations)
            self.print_debug("new_relations: %s" % (new_relations, ))
            # check the total number of new relations
            if len(new_relations) != num_new_relations:
                continue
            # check sanity of relations
            forward = dict(new_relations)
            if len(forward) != num_new_relations:
                continue
            reverse = dict((j, i) for i, j in new_relations)
            if len(reverse) != num_new_relations:
                continue
            # check the constraints
            for a0, b0 in constraints0:
                if forward[a0] not in subject_graph.neighbors[forward[b0]]:
                    forward = None
                    break
            if forward is None:
                continue
            yield forward