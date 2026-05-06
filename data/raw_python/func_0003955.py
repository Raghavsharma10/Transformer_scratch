def distances(self):
        """The matrix with the all-pairs shortest path lenghts"""
        from molmod.ext import graphs_floyd_warshall
        distances = np.zeros((self.num_vertices,)*2, dtype=int)
        #distances[:] = -1 # set all -1, which is just a very big integer
        #distances.ravel()[::len(distances)+1] = 0 # set diagonal to zero
        for i, j in self.edges: # set edges to one
            distances[i, j] = 1
            distances[j, i] = 1
        graphs_floyd_warshall(distances)
        return distances