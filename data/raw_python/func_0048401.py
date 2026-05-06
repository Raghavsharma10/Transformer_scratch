def get_coord_line_number(self,coord):
    """return the one-indexed line number given the coordinates"""
    if coord[0] in self._coords:
      if coord[1] in self._coords[coord[0]]:
        return self._coords[coord[0]][coord[1]]
    return None