def equivalent_vertices(self):
        """A dictionary with symmetrically equivalent vertices."""
        level1 = {}
        for i, row in enumerate(self.vertex_fingerprints):
            key = row.tobytes()
            l = level1.get(key)
            if l is None:
                l = set([i])
                level1[key] = l
            else:
                l.add(i)
        level2 = {}
        for key, vertices in level1.items():
            for vertex in vertices:
                level2[vertex] = vertices
        return level2