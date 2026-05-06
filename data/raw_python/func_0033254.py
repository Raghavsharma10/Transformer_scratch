def crop(self, lat, lon, var):
        """ Crop a subset of the dataset for each var

            Given doy, depth, lat and lon, it returns the smallest subset
              that still contains the requested coordinates inside it.

            It handels special cases like a region around greenwich and
            the international date line.

            Accepts 0 to 360 and -180 to 180 longitude reference.

            It extends time and longitude coordinates, so simplify the use
               of series. For example, a ship track can be requested with
               a longitude sequence like [352, 358, 364, 369, 380].
        """
        dims, idx = cropIndices(self.dims, lat, lon)
        subset = {}
        for v in var:
            subset = {v: self.ncs[0][v][idx['yn'], idx['xn']]}
        return subset, dims