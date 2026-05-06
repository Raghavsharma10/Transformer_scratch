def check_next_match(self, match, new_relations, subject_graph, one_match):
        """Check if the (onset for a) match can be a valid (part of a) ring"""
        if not CustomPattern.check_next_match(self, match, new_relations, subject_graph, one_match):
            return False
        if self.strong:
            # can this ever become a strong ring?
            vertex1_start = match.forward[self.pattern_graph.central_vertex]
            for vertex1 in new_relations.values():
                paths = list(subject_graph.iter_shortest_paths(vertex1, vertex1_start))
                if self.size % 2 == 0 and len(match) == self.size:
                    if len(paths) != 2:
                        #print "NRingPattern.check_next_match: not strong a.1"
                        return False
                    for path in paths:
                        if len(path) != len(match)//2+1:
                            #print "NRingPattern.check_next_match: not strong a.2"
                            return False
                else:
                    if len(paths) != 1:
                        #print "NRingPattern.check_next_match: not strong b.1"
                        return False
                    if len(paths[0]) != (len(match)+1)//2:
                        #print "NRingPattern.check_next_match: not strong b.2"
                        return False
            #print "RingPattern.check_next_match: no remarks"
        return True