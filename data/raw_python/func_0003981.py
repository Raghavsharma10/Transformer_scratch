def check_next_match(self, match, new_relations, subject_graph, one_match):
        """Check if the (onset for a) match can be a valid"""
        # only returns true for ecaxtly one set of new_relations from all the
        # ones that are symmetrically equivalent
        if not (self.criteria_sets is None or one_match):
            for check in self.duplicate_checks:
                vertex_a = new_relations.get(check[0])
                vertex_b = new_relations.get(check[1])
                if vertex_a is None and vertex_b is None:
                    continue # if this pair is completely absent in the new
                    # relations, it is either completely in the match or it
                    # is to be matched. So it is either already checked for
                    # symmetry duplicates, or it will be check in future.
                if vertex_a is None:
                    # maybe vertex_a is in the match and vertex_b is the only
                    # one in the new relations. try to get vertex_a from the
                    # match.
                    vertex_a = match.forward.get(check[0])
                    if vertex_a is None:
                        # ok, vertex_a is to be found, don't care about it right
                        # now. it will be checked in future calls.
                        continue
                elif vertex_b is None:
                    # maybe vertex_b is in the match and vertex_a is the only
                    # one in the new relations. try to get vertex_b from the
                    # match.
                    vertex_b = match.forward.get(check[1])
                    if vertex_b is None:
                        # ok, vertex_b is to be found, don't care about it right
                        # now. it will be checked in future calls.
                        continue
                if vertex_a > vertex_b:
                    # Why does this work? The answer is not so easy to explain,
                    # and certainly not easy to find. if vertex_a > vertex_b, it
                    # means that there is a symmetry operation that leads to
                    # an equivalent match where vertex_b < vertex_a. The latter
                    # match is preferred for as much pairs (vertex_a, vertex_b)
                    # as possible without rejecting all possible matches. The
                    # real difficulty is to construct a proper list of
                    # (vertex_a, vertex_b) pairs that will reject all but one
                    # matches. I conjecture that this list contains all the
                    # first two vertices from each normalized symmetry cycle of
                    # the pattern graph. We need a math guy to do the proof. -- Toon
                    return False
            return True
        return True