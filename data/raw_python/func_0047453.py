def adjacent(self,rng2):
    """ Test for adjacency.  

    :param rng2:
    :param use_direction: false by default
    :param type: GenomicRange
    :param type: use_direction
    """
    if self.chr != rng2.chr: return False
    if self.direction != rng2.direction and use_direction: return False
    if self.end == rng2.start-1:  return True
    if self.start-1 == rng2.end: return True
    return False