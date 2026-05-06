def max(self):
        """ -> #float :func:numpy.max of the timing intervals """
        return round(np.max(self.array), self.precision)\
            if len(self.array) else None