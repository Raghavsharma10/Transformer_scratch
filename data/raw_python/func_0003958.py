def fingerprint(self):
        """A total graph fingerprint

           The result is invariant under permutation of the vertex indexes. The
           chance that two different (molecular) graphs yield the same
           fingerprint is small but not zero. (See unit tests.)"""
        if self.num_vertices == 0:
            return np.zeros(20, np.ubyte)
        else:
            return sum(self.vertex_fingerprints)