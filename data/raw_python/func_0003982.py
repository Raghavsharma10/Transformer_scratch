def iter_final_matches(self, canonical_match, subject_graph, one_match):
        """Given a match, iterate over all related equivalent matches

           When criteria sets are defined, the iterator runs over all symmetric
           equivalent matches that fulfill one of the criteria sets. When not
           criteria sets are defined, the iterator only yields the input match.
        """
        if self.criteria_sets is None or one_match:
            yield canonical_match
        else:
            for criteria_set in self.criteria_sets:
                satisfied_match_tags = set([])
                for symmetry in self.pattern_graph.symmetries:
                    final_match = canonical_match * symmetry
                    #print final_match
                    if criteria_set.test_match(final_match, self.pattern_graph, subject_graph):
                        match_tags = tuple(
                            self.vertex_tags.get(symmetry.reverse[vertex0])
                            for vertex0
                            in range(self.pattern_graph.num_vertices)
                        )
                        if match_tags not in satisfied_match_tags:
                            final_match.__dict__.update(criteria_set.info)
                            yield final_match
                            satisfied_match_tags.add(match_tags)