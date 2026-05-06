def intersect_4(self, second, third, fourth):
    """
     Intersection routine for four inputs.
    """
    self.intersection(second)
    self.intersection(third)
    self.intersection(fourth)
    self.coalesce()
    return len(self)