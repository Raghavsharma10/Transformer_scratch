def get_vertex_fingerprints(self, vertex_strings, edge_strings, num_iter=None):
        """Return an array with fingerprints for each vertex"""
        import hashlib
        def str2array(x):
            """convert a hash string to a numpy array of bytes"""
            if len(x) == 0:
                return np.zeros(0, np.ubyte)
            elif sys.version_info[0] == 2:
                return np.frombuffer(x, np.ubyte)
            else:
                return np.frombuffer(x.encode(), np.ubyte)
        hashrow = lambda x: np.frombuffer(hashlib.sha1(x.data).digest(), np.ubyte)
        # initialization
        result = np.zeros((self.num_vertices, 20), np.ubyte)
        for i in range(self.num_vertices):
            result[i] = hashrow(str2array(vertex_strings[i]))
        for i in range(self.num_edges):
            a, b = self.edges[i]
            tmp = hashrow(str2array(edge_strings[i]))
            result[a] += tmp
            result[b] += tmp
        work = result.copy()
        # iterations
        if num_iter is None:
            num_iter = self.max_distance
        for i in range(num_iter):
            for a, b in self.edges:
                work[a] += result[b]
                work[b] += result[a]
            #for a in xrange(self.num_vertices):
            #    for b in xrange(self.num_vertices):
            #        work[a] += hashrow(result[b]*self.distances[a, b])
            for a in range(self.num_vertices):
                result[a] = hashrow(work[a])
        return result