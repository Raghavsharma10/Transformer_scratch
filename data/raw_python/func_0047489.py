def get_junctions_string(self):
    """Get a string representation of the junctions.  This is almost identical to a previous function.

    :return: string representation of junction
    :rtype: string
    """
    self._initialize()
    return ';'.join([x.get_range_string() for x in self.junctions])