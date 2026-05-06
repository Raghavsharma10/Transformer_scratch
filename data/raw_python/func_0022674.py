def _find_cut_triangle(self, edge):
        """
        Return the triangle that has edge[0] as one of its vertices and is 
        bisected by edge.
        
        Return None if no triangle is found.
        """
        edges = []  # opposite edge for each triangle attached to edge[0]
        for tri in self.tris:
            if edge[0] in tri:
                edges.append(self._edge_opposite_point(tri, edge[0]))
                
        for oedge in edges:
            o1 = self._orientation(edge, oedge[0])
            o2 = self._orientation(edge, oedge[1]) 
            #debug(edge, oedge, o1, o2)
            #debug(self.pts[np.array(edge)])
            #debug(self.pts[np.array(oedge)])
            if o1 != o2:
                return (edge[0], oedge[0], oedge[1])
        
        return None