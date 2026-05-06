def central_vertices(self):
        """Vertices that have the lowest maximum distance to any other vertex"""
        max_distances = self.distances.max(0)
        max_distances_min = max_distances[max_distances > 0].min()
        return (max_distances == max_distances_min).nonzero()[0]