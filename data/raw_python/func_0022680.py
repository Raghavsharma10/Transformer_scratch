def _edge_below_front(self, edge, front_index):
        """Return True if *edge* is below the current front. 
        
        One of the points in *edge* must be _on_ the front, at *front_index*.
        """
        f0 = self._front[front_index-1]
        f1 = self._front[front_index+1]
        return (self._orientation(edge, f0) > 0 and 
                self._orientation(edge, f1) < 0)