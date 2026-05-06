def has_vertex_color(self):
        """Return True if this data set has vertex color information"""
        for v in (self._vertex_colors, self._vertex_colors_indexed_by_faces,
                  self._vertex_colors_indexed_by_edges):
            if v is not None:
                return True
        return False