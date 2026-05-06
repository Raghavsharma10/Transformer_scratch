def _mk_connectivity_flats(self, i12, j1, j2, mat_data, flats, elev, mag):
        """
        Helper function for _mk_adjacency_matrix. This calcualtes the
        connectivity for flat regions. Every pixel in the flat will drain
        to a random pixel in the flat. This accumulates all the area in the
        flat region to a single pixel. All that area is then drained from
        that pixel to the surroundings on the flat. If the border of the
        flat has a single pixel with a much lower elevation, all the area will
        go towards that pixel. If the border has pixels with similar elevation,
        then the area will be distributed amongst all the border pixels
        proportional to their elevation.
        """
        nn, mm = flats.shape
        NN = np.prod(flats.shape)
        # Label the flats
        assigned, n_flats = spndi.label(flats, FLATS_KERNEL3)

        flat_ids, flat_coords, flat_labelsf = _get_flat_ids(assigned)
        flat_j = [None] * n_flats
        flat_prop = [None] * n_flats
        flat_i = [None] * n_flats

        # Temporary array to find the flats
        edges = np.zeros_like(flats)
        # %% Calcute the flat drainage
        warn_flats = []
        for ii in xrange(n_flats):
            ids_flats = flat_ids[flat_coords[ii]:flat_coords[ii+1]]
            edges[:] = 0
            j = ids_flats % mm
            i = ids_flats // mm
            for iii in [-1, 0, 1]:
                for jjj in [-1, 0, 1]:
                    i_2 = i + iii
                    j_2 = j + jjj
                    ids_tmp = (i_2 >= 0) & (j_2 >= 0) & (i_2 < nn) & (j_2 < mm)
                    edges[i_2[ids_tmp], j_2[ids_tmp]] += \
                        FLATS_KERNEL3[iii+1, jjj+1]
            edges.ravel()[ids_flats] = 0
            ids_edge = np.argwhere(edges.ravel()).squeeze()

            flat_elev_loc = elev.ravel()[ids_flats]
            # It is possble for the edges to merge 2 flats, so we need to
            # take the lower elevation to avoid large circular regions
            flat_elev = flat_elev_loc.min()

            loc_elev = elev.ravel()[ids_edge]
            # Filter out any elevations larger than the flat elevation
            # TODO: Figure out if this should be <= or <
            I_filt = loc_elev < flat_elev
            try:
                loc_elev = loc_elev[I_filt]
                loc_slope = mag.ravel()[ids_edge][I_filt]
            except: # If this is fully masked out (i.e. inside a no-data area)
                loc_elev = np.array([])
                loc_slope = np.array([])

            loc_dx = self.dX.mean()

            # Now I have to figure out if I should just use the minimum or
            # distribute amongst many pixels on the flat boundary
            n = len(loc_slope)
            if n == 0:  # Flat does not have anywhere to drain
                # Let's see if the flat goes to the edge. If yes, we'll just
                # distribute the area along the edge.
                ids_flat_on_edge = ((ids_flats % mag.shape[1]) == 0) | \
                    ((ids_flats % mag.shape[1]) == (mag.shape[1] - 1)) | \
                    (ids_flats <= mag.shape[1]) | \
                    (ids_flats >= (mag.shape[1] * (mag.shape[0] - 1)))
                if ids_flat_on_edge.sum() == 0:
                    warn_flats.append(ii)
                    continue

                drain_ids = ids_flats[ids_flat_on_edge]
                loc_proportions = mag.ravel()[ids_flats[ids_flat_on_edge]]
                loc_proportions /= loc_proportions.sum()

                ids_flats = ids_flats[~ids_flat_on_edge]
                # This flat is entirely on the edge of the image
                if len(ids_flats) == 0:
                    # therefore, whatever drains into it is done.
                    continue
                flat_elev_loc = flat_elev_loc[~ids_flat_on_edge]
            else:  # Flat has a place to drain to
                min_edges = np.zeros(loc_slope.shape, bool)
                min_edges[np.argmin(loc_slope)] = True
                # Add to the min edges any edge that is within an error
                # tolerance as small as the minimum
                min_edges = (loc_slope + loc_slope * loc_dx / 2) \
                    >= loc_slope[min_edges]

                drain_ids = ids_edge[I_filt][min_edges]

                loc_proportions = loc_slope[min_edges]
                loc_proportions /= loc_proportions.sum()

            # Now distribute the connectivity amongst the chosen elevations
            # proportional to their slopes

            # First, let all the the ids in the flats drain to 1
            # flat id (for ease)
            one_id = np.zeros(ids_flats.size, bool)
            one_id[np.argmin(flat_elev_loc)] = True

            j1.ravel()[ids_flats[~one_id]] = ids_flats[one_id]
            mat_data.ravel()[ids_flats[~one_id]] = 1
            # Negative indices will be eliminated before making the matix
            j2.ravel()[ids_flats[~one_id]] = -1
            mat_data.ravel()[ids_flats[~one_id] + NN] = 0

            # Now drain the 1 flat to the drains
            j1.ravel()[ids_flats[one_id]] = drain_ids[0]
            mat_data.ravel()[ids_flats[one_id]] = loc_proportions[0]
            if len(drain_ids) > 1:
                j2.ravel()[ids_flats[one_id]] = drain_ids[1]
                mat_data.ravel()[ids_flats[one_id] + NN] = loc_proportions[1]

            if len(loc_proportions > 2):
                flat_j[ii] = drain_ids[2:]
                flat_prop[ii] = loc_proportions[2:]
                flat_i[ii] = np.ones(drain_ids[2:].size, 'int64') * ids_flats[one_id]
        try:
            flat_j = np.concatenate([fj for fj in flat_j if fj is not None])
            flat_prop = \
                np.concatenate([fp for fp in flat_prop if fp is not None])
            flat_i = np.concatenate([fi for fi in flat_i if fi is not None])
        except:
            flat_j = np.array([], 'int64')
            flat_prop = np.array([], 'float64')
            flat_i = np.array([], 'int64')

        if len(warn_flats) > 0:
            warnings.warn("Warning %d flats had no place" % len(warn_flats) +
                          " to drain to --> these are pits (check pit-remove"
                          "algorithm).")
        return j1, j2, mat_data, flat_i, flat_j, flat_prop