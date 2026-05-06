def min(self):
        """ -> #float :func:numpy.min of the timing intervals """
        return round(np.min(self.array), self.precision)\
            if len(self.array) else None