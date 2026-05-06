def pointlist(points, sr):
    """Convert a list of the form [[x, y] ...] to a list of Point instances
       with the given x, y coordinates."""
    assert all(isinstance(pt, Point) or len(pt) == 2 
               for pt in points), "Point(s) not in [x, y] form"
    return [coord if isinstance(coord, Point) 
                  else Point(coord[0], coord[1], sr)
            for coord in points]