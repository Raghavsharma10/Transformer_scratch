def complete(self, match, subject_graph):
        """Check the completeness of the ring match"""
        if not CustomPattern.complete(self, match, subject_graph):
            return False
        if self.strong:
            # If the ring is not strong, return False
            if self.size % 2 == 0:
                # even ring
                for i in range(self.size//2):
                    vertex1_start = match.forward[i]
                    vertex1_stop = match.forward[(i+self.size//2)%self.size]
                    paths = list(subject_graph.iter_shortest_paths(vertex1_start, vertex1_stop))
                    if len(paths) != 2:
                        #print "Even ring must have two paths between opposite vertices"
                        return False
                    for path in paths:
                        if len(path) != self.size//2+1:
                            #print "Paths between opposite vertices must half the size of the ring+1"
                            return False
            else:
                # odd ring
                for i in range(self.size//2+1):
                    vertex1_start = match.forward[i]
                    vertex1_stop = match.forward[(i+self.size//2)%self.size]
                    paths = list(subject_graph.iter_shortest_paths(vertex1_start, vertex1_stop))
                    if len(paths) > 1:
                        return False
                    if len(paths[0]) != self.size//2+1:
                        return False
                    vertex1_stop = match.forward[(i+self.size//2+1)%self.size]
                    paths = list(subject_graph.iter_shortest_paths(vertex1_start, vertex1_stop))
                    if len(paths) > 1:
                        return False
                    if len(paths[0]) != self.size//2+1:
                        return False
        return True