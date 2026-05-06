def median(self):
        """ -> #float :func:numpy.median of the timing intervals """
        return round(float(np.median(self.array)), self.precision)\
            if len(self.array) else None