def intersect_3(self, second, third):
    """
    Intersection routine for three inputs.  Built out of the intersect,
    coalesce and play routines
    """
    self.intersection(second)
    self.intersection(third)
    self.coalesce()
    return len(self)