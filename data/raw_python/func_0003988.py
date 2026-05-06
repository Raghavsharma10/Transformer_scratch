def check_next_match(self, match, new_relations, subject_graph, one_match):
        """Check if the (onset for a) match can be a valid (part of a) ring"""
        # avoid duplicate rings (order of traversal)
        if len(match) == 3:
            if match.forward[1] < match.forward[2]:
                #print "RingPattern.check_next_match: duplicate order", match.forward[1], match.forward[2]
                return False
        # avoid duplicate rings (starting point)
        for vertex1 in new_relations.values():
            if vertex1 < match.forward[0]:
                #print "RingPattern.check_next_match: duplicate start", vertex1, match.forward[0]
                return False
        # can this ever become a strong ring?
        for vertex1 in new_relations.values():
            paths = list(subject_graph.iter_shortest_paths(vertex1, match.forward[0]))
            if len(paths) != 1:
                #print "RingPattern.check_next_match: not strong 1"
                return False
            if len(paths[0]) != (len(match)+1)//2:
                #print "RingPattern.check_next_match: not strong 2"
                return False
        return True