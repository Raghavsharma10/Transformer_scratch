def interpolate(self, doy, depth, lat, lon, var):
        """ Interpolate each var on the coordinates requested

        """
        subset, dims = self.crop(doy, depth, lat, lon, var)

        # Subset contains everything requested. No need to interpolate.
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

        output = {}
        for v in var:
            output[v] = ma.masked_all(
                    (doy.size, depth.size, lat.size, lon.size),
                    dtype=subset[v].dtype)

            # These interpolators don't understand Masked Arrays, but do NaN
            if subset[v].dtype in ['int32']:
                subset[v] = subset[v].astype('f')
            subset[v][ma.getmaskarray(subset[v])] = np.nan
            subset[v] = subset[v].data

        # First linear interpolate on time.
        if not (doy == dims['time']).all():
            for v in subset.keys():
                f = interp1d(dims['time'], subset[v], axis=0)
                subset[v] = f(doy)
            dims['time'] = np.atleast_1d(doy)

        if not (np.all(lat == dims['lat']) and np.all(lon == dims['lon'])):
            # Lat x Lon target coordinates are the same for all time and depth.
            points_out = []
            for latn in lat:
                for lonn in lon:
                    points_out.append([latn, lonn])
            points_out = np.array(points_out)

            # Interpolate on X/Y plane
            for v in subset:
                tmp = np.nan * np.ones(
                        (doy.size, dims['depth'].size, lat.size, lon.size),
                        dtype=subset[v].dtype)
                for nt in range(doy.size):
                    for nz in range(dims['depth'].size):
                        data = subset[v][nt, nz]
                        # The valid data
                        idx = np.nonzero(~np.isnan(data))
                        if idx[0].size > 0:
                            points = np.array([
                                dims['lat'][idx[0]], dims['lon'][idx[1]]]).T
                            values = data[idx]
                            # Interpolate along the dimensions that have more than
                            #   one position, otherwise it means that the output
                            #   is exactly on that coordinate.
                            #ind = np.array([np.unique(points[:, i]).size > 1
                            #    for i in range(points.shape[1])])
                            #assert ind.any()

                            try:
                                values_out = griddata(
                                    #np.atleast_1d(np.squeeze(points[:, ind])),
                                    np.atleast_1d(np.squeeze(points)),
                                    values,
                                    #np.atleast_1d(np.squeeze(points_out[:, ind])))
                                    np.atleast_1d(np.squeeze(points_out)))
                            except:
                                values_out = []
                                for p in points_out:
                                    try:
                                        values_out.append(griddata(
                                            np.atleast_1d(np.squeeze(points)),
                                            values,
                                            np.atleast_1d(np.squeeze(
                                                p))))
                                    except:
                                        values_out.append(np.nan)
                                values_out = np.array(values_out)

                            # Remap the interpolated value back into a 4D array
                            idx = np.isfinite(values_out)
                            for [y, x], out in zip(
                                    points_out[idx], values_out[idx]):
                                tmp[nt, nz, y==lat, x==lon] = out
                subset[v] = tmp

        # Interpolate on z
        same_depth = (np.shape(depth) == dims['depth'].shape) and \
                np.allclose(depth, dims['depth'])
        if not same_depth:
            for v in list(subset.keys()):
                try:
                    f = interp1d(dims['depth'], subset[v], axis=1, bounds_error=False)
                    # interp1d does not handle Masked Arrays
                    subset[v] = f(np.array(depth))
                except:
                    print("Fail to interpolate '%s' in depth" % v)
                    del(subset[v])

        for v in subset:
            if output[v].dtype in ['int32']:
                subset[v] = np.round(subset[v])
            output[v][:] = ma.fix_invalid(subset[v][:])

        return output