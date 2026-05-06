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
        dims, idx = cropIndices(self.dims, lat, lon, depth)

        dims['time'] = np.atleast_1d(doy)
        idx['tn'] = np.arange(dims['time'].size)

        # Temporary solution. Create an object for CARS dataset
        xn = idx['xn']
        yn = idx['yn']
        zn = idx['zn']
        tn = idx['tn']

        subset = {}
        for v in var:
            if v == 'mn':
                mn = []
                for d in doy:
                    t = 2 * np.pi * d/366
                    # Naive solution
                    # FIXME: This is not an efficient solution.
                    value = self.ncs[0]['mean'][:, yn, xn]
                    value[:64] += self.ncs[0]['an_cos'][:, yn, xn] * np.cos(t) + \
                        self.ncs[0]['an_sin'][:, yn, xn] * np.sin(t)
                    value[:55] += self.ncs[0]['sa_cos'][:, yn, xn] * np.cos(2*t) + \
                        self.ncs[0]['sa_sin'][:, yn, xn] * np.sin(2*t)
                    mn.append(value[zn])

                subset['mn'] = ma.asanyarray(mn)
            else:
                subset[v] = ma.asanyarray(
                        doy.size * [self[v][zn, yn, xn]])
        return subset, dims