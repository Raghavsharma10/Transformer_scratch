def get_radius_ranges(self, radius, mic=False):
        """Return ranges of indexes of the interacting neighboring unit cells

           Interacting neighboring unit cells have at least one point in their
           box volume that has a distance smaller or equal than radius to at
           least one point in the central cell. This concept is of importance
           when computing pair wise long-range interactions in periodic systems.

           The mic (stands for minimum image convention) option can be used to
           change the behavior of this routine such that only neighboring cells
           are considered that have at least one point withing a distance below
           `radius` from the center of the reference cell.
        """
        result = np.zeros(3, int)
        for i in range(3):
            if self.spacings[i] > 0:
                if mic:
                    result[i] = np.ceil(radius/self.spacings[i]-0.5)
                else:
                    result[i] = np.ceil(radius/self.spacings[i])
        return result