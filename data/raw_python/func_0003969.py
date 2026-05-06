def get_part(self, vertex_in, vertices_border):
        """List all vertices that are connected to vertex_in, but are not
           included in or 'behind' vertices_border.
        """
        vertices_new = set(self.neighbors[vertex_in])
        vertices_part = set([vertex_in])

        while len(vertices_new) > 0:
            pivot = vertices_new.pop()
            if pivot in vertices_border:
                continue
            vertices_part.add(pivot)
            pivot_neighbors = set(self.neighbors[pivot])
            pivot_neighbors -= vertices_part
            vertices_new |= pivot_neighbors

        return vertices_part