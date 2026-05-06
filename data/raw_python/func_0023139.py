def get_edges(self, indexed=None):
        """Edges of the mesh
        
        Parameters
        ----------
        indexed : str | None
           If indexed is None, return (Nf, 3) array of vertex indices,
           two per edge in the mesh.
           If indexed is 'faces', then return (Nf, 3, 2) array of vertex
           indices with 3 edges per face, and two vertices per edge.

        Returns
        -------
        edges : ndarray
            The edges.
        """
        
        if indexed is None:
            if self._edges is None:
                self._compute_edges(indexed=None)
            return self._edges
        elif indexed == 'faces':
            if self._edges_indexed_by_faces is None:
                self._compute_edges(indexed='faces')
            return self._edges_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")