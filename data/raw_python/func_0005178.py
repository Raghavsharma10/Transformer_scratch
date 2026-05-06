def mean(self):
        """ -> #float :func:numpy.mean of the timing intervals """
        return round(np.mean(self.array), self.precision)\
            if len(self.array) else None