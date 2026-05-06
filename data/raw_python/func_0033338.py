def interpolate(self, doy, depth, lat, lon, var):
        """ Interpolate each var on the coordinates requested

        """

        subset, dims = self.crop(doy, depth, lat, lon, var)

        if np.all([d in dims['time'] for d in doy]) & \
                np.all([z in dims['depth'] for z in depth]) & \
                np.all([y in dims['lat'] for y in lat]) & \
                np.all([x in dims['lon'] for x in lon]):
                    dn = np.nonzero([d in doy for d in dims['time']])[0]
                    zn = np.nonzero([z in depth for z in dims['depth']])[0]
                    yn = np.nonzero([y in lat for y in dims['lat']])[0]
                    xn = np.nonzero([x in lon for x in dims['lon']])[0]
                    output = {}
                    for v in subset:
                        # output[v] = subset[v][dn, zn, yn, xn]
                        # Seriously that this is the way to do it?!!??
                        output[v] = subset[v][:, :, :, xn][:, :, yn][:, zn][dn]
                    return output

        # The output coordinates shall be created only once.
        points_out = []
        for doyn in doy:
            for depthn in depth:
                for latn in lat:
                    for lonn in lon:
                        points_out.append([doyn, depthn, latn, lonn])
        points_out = np.array(points_out)

        output = {}
        for v in var:
            output[v] = ma.masked_all(
                    (doy.size, depth.size, lat.size, lon.size),
                    dtype=subset[v].dtype)

            # The valid data
            idx = np.nonzero(~ma.getmaskarray(subset[v]))

            if idx[0].size > 0:
                points = np.array([
                    dims['time'][idx[0]], dims['depth'][idx[1]],
                    dims['lat'][idx[2]], dims['lon'][idx[3]]]).T
                values = subset[v][idx]

                # Interpolate along the dimensions that have more than one
                #   position, otherwise it means that the output is exactly
                #   on that coordinate.
                ind = np.array(
                        [np.unique(points[:, i]).size > 1 for i in
                            range(points.shape[1])])
                assert ind.any()

                # These interpolators understand NaN, but not masks.
                values[ma.getmaskarray(values)] = np.nan

                values_out = griddata(
                        np.atleast_1d(np.squeeze(points[:, ind])),
                        values,
                        np.atleast_1d(np.squeeze(points_out[:, ind]))
                        )

                # Remap the interpolated value back into a 4D array
                idx = np.isfinite(values_out)
                for [t, z, y, x], out in zip(points_out[idx], values_out[idx]):
                    output[v][t==doy, z==depth, y==lat, x==lon] = out

            output[v] = ma.fix_invalid(output[v])

        return output