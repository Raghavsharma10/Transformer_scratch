def count_overlap(self, time, other_object, other_time):
        """
        Counts the number of points that overlap between this STObject and another STObject. Used for tracking.
        """
        ti = np.where(time == self.times)[0][0]
        ma = np.where(self.masks[ti].ravel() == 1)
        oti = np.where(other_time == other_object.times)[0]
        obj_coords = np.zeros(self.masks[ti].sum(), dtype=[('x', int), ('y', int)])
        other_obj_coords = np.zeros(other_object.masks[oti].sum(), dtype=[('x', int), ('y', int)])
        obj_coords['x'] = self.i[ti].ravel()[ma]
        obj_coords['y'] = self.j[ti].ravel()[ma]
        other_obj_coords['x'] = other_object.i[oti][other_object.masks[oti] == 1]
        other_obj_coords['y'] = other_object.j[oti][other_object.masks[oti] == 1]
        return float(np.intersect1d(obj_coords,
                                    other_obj_coords).size) / np.maximum(self.masks[ti].sum(),
                                                                         other_object.masks[oti].sum())