def stdev(self):
        """ -> #float :func:numpy.std of the timing intervals """
        return round(np.std(self.array), self.precision)\
            if len(self.array) else None