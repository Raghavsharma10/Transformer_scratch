def crop(self, doy, depth, lat, lon, var):
        """ Crop a subset of the dataset for each var

            Given doy, depth, lat and lon, it returns the smallest subset
              that still contains the requested coordinates inside it.

            It handels special cases like a region around greenwich and
            the international date line.

            Accepts 0 to 360 and -180 to 180 longitude reference.

            It extends time and longitude coordinates, so simplify the use
               of series. For example, a ship track can be requested with
               a longitude sequence like [352, 358, 364, 369, 380], and
               the equivalent for day of year above 365.
        """
        dims, idx = cropIndices(self.dims, lat, lon, depth, doy)
        subset = {}
        for v in var:
            subset[v] = ma.asanyarray([
                self.ncs[tnn][v][0, idx['zn'], idx['yn'], idx['xn']] \
                        for tnn in idx['tn']])
        return subset, dims