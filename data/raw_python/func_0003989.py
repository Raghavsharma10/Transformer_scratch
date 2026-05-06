def complete(self, match, subject_graph):
        """Check the completeness of a ring match"""
        size = len(match)
        # check whether we have an odd strong ring
        if match.forward[size-1] in subject_graph.neighbors[match.forward[size-2]]:
            # we have an odd closed cycle. check if this is a strong ring
            order = list(range(0, size, 2)) + list(range(1, size-1, 2))[::-1]
            ok = True
            for i in range(len(order)//2):
                # Count the number of paths between two opposite points in the
                # ring. Since the ring has an odd number of vertices, each
                # vertex has two semi-opposite vertices.
                count = len(list(subject_graph.iter_shortest_paths(
                    match.forward[order[i]],
                    match.forward[order[(i+size//2)%size]]
                )))
                if count > 1:
                    ok = False
                    break
                count = len(list(subject_graph.iter_shortest_paths(
                    match.forward[order[i]],
                    match.forward[order[(i+size//2+1)%size]]
                )))
                if count > 1:
                    ok = False
                    break
            if ok:
                match.ring_vertices = tuple(match.forward[i] for i in order)
                #print "RingPattern.complete: found odd ring"
                return True
            #print "RingPattern.complete: no odd ring"
        # check whether we have an even strong ring
        paths = list(subject_graph.iter_shortest_paths(
            match.forward[size-1],
            match.forward[size-2]
        ))
        #print "RingPattern.complete: even paths", paths
        if (size > 3 and len(paths) == 1 and len(paths[0]) == 3) or \
           (size == 3 and len(paths) == 2 and len(paths[0]) == 3):
            path = paths[0]
            if size == 3 and path[1] == match.forward[0]:
                path = paths[1]
            # we have an even closed cycle. check if this is a strong ring
            match.add_relation(size, path[1])
            size += 1
            order = list(range(0, size, 2)) + list(range(size-1, 0, -2))
            ok = True
            for i in range(len(order)//2):
                count = len(list(subject_graph.iter_shortest_paths(
                    match.forward[order[i]],
                    match.forward[order[(i+size//2)%size]]
                )))
                if count != 2:
                    ok = False
                    break
            if ok:
                # also check if this does not violate the requirement for a
                # unique origin:
                if match.forward[size-1] < match.forward[0]:
                    ok = False
            if not ok:
                vertex1 = match.forward[size-1]
                del match.forward[size-1]
                del match.reverse[vertex1]
                size -= 1
                #print "RingPattern.complete: no even ring"
            else:
                match.ring_vertices = tuple(match.forward[i] for i in order)
                #print "RingPattern.complete: found even ring"
            return ok
        #print "RingPattern.complete: not at all"
        return False