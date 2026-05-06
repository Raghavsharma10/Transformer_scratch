def independent_vertices(self):
        """Lists of vertices that are only interconnected within each list

           This means that there is no path from a vertex in one list to a
           vertex in another list. In case of a molecular graph, this would
           yield the atoms that belong to individual molecules.
        """
        candidates = set(range(self.num_vertices))

        result = []
        while len(candidates) > 0:
            pivot = candidates.pop()
            group = [
                vertex for vertex, distance
                in self.iter_breadth_first(pivot)
            ]
            candidates.difference_update(group)

            # this sort makes sure that the order of the vertices is respected
            group.sort()
            result.append(group)
        return result