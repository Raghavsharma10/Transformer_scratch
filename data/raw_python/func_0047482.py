def avg_mutual_coverage(self,gpd):
    """get the coverage fraction of each transcript then return the geometric mean

    :param gpd: Another transcript
    :type gpd: Transcript
    :return: avg_coverage
    :rtype: float
    """
    ov = self.overlap_size(gpd)
    if ov == 0: return 0
    xfrac = float(ov) / float(self.get_length())
    yfrac = float(ov) / float(gpd.get_length())
    return sqrt(xfrac*yfrac)