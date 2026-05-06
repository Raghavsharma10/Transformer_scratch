def interpolate(self, lat, lon, var):
        """ Interpolate each var on the coordinates requested

        """

        subset, dims = self.crop(lat, lon, var)

        if np.all([y in dims['lat'] for y in lat]) & \
                np.all([x in dims['lon'] for x in lon]):
                    yn = np.nonzero([y in lat for y in dims['lat']])[0]
                    xn = np.nonzero([x in lon for x in dims['lon']])[0]
                    output = {}
                    for v in subset:
                        # output[v] = subset[v][dn, zn, yn, xn]
                        # Seriously that this is the way to do it?!!??
                        output[v] = subset[v][:, xn][yn]
                    return output

        # The output coordinates shall be created only once.
        points_out = []
        for latn in lat:
            for lonn in lon:
                points_out.append([latn, lonn])
        points_out = np.array(points_out)

        output = {}
        for v in var:
            output[v] = ma.masked_all(
                    (lat.size, lon.size),
                    dtype=subset[v].dtype)

            # The valid data
            idx = np.nonzero(~ma.getmaskarray(subset[v]))

            if idx[0].size > 0:
                points = np.array([
                    dims['lat'][idx[0]], dims['lon'][idx[1]]]).T
                values = subset[v][idx]

                # Interpolate along the dimensions that have more than one
                #   position, otherwise it means that the output is exactly
                #   on that coordinate.
                ind = np.array(
                        [np.unique(points[:, i]).size > 1 for i in
                            range(points.shape[1])])
                assert ind.any()

                values_out = griddata(
                        np.atleast_1d(np.squeeze(points[:, ind])),
                        values,
                        np.atleast_1d(np.squeeze(points_out[:, ind]))
                        )

                # Remap the interpolated value back into a 4D array
                idx = np.isfinite(values_out)
                for [y, x], out in zip(points_out[idx], values_out[idx]):
                    output[v][y==lat, x==lon] = out

        return output