def _mk_connectivity_pits(self, i12, flats, elev, mag, dX, dY):
        """
        Helper function for _mk_adjacency_matrix. This is a more general
        version of _mk_adjacency_flats which drains pits and flats to nearby
        but non-adjacent pixels. The slope magnitude (and flats mask) is
        updated for these pits and flats so that the TWI can be computed.
        """
        
        e = elev.data.ravel()

        pit_i = []
        pit_j = []
        pit_prop = []
        warn_pits = []
        
        pits = i12[flats & (elev > 0)]
        I = np.argsort(e[pits])
        for pit in pits[I]:
            # find drains
            pit_area = np.array([pit], 'int64')

            drain = None
            epit = e[pit]
            for it in range(self.drain_pits_max_iter):
                border = get_border_index(pit_area, elev.shape, elev.size)

                eborder = e[border]
                emin = eborder.min()
                if emin < epit:
                    drain = border[eborder < epit]
                    break

                pit_area = np.concatenate([pit_area, border[eborder == emin]])

            if drain is None:
                warn_pits.append(pit)
                continue
            
            ipit, jpit = np.unravel_index(pit, elev.shape)
            Idrain, Jdrain = np.unravel_index(drain, elev.shape)

            # filter by drain distance in coordinate space
            if self.drain_pits_max_dist:
                dij = np.sqrt((ipit - Idrain)**2 + (jpit-Jdrain)**2)
                b = dij <= self.drain_pits_max_dist
                if not b.any():
                    warn_pits.append(pit)
                    continue
                drain = drain[b]
                Idrain = Idrain[b]
                Jdrain = Jdrain[b]
            
            # calculate real distances
            dx = [_get_dX_mean(dX, ipit, idrain) * (jpit - jdrain)
                  for idrain, jdrain in zip(Idrain, Jdrain)]
            dy = [dY[make_slice(ipit, idrain)].sum() for idrain in Idrain]
            dxy = np.sqrt(np.array(dx)**2 + np.array(dy)**2)

            # filter by drain distance in real space
            if self.drain_pits_max_dist_XY:
                b = dxy <= self.drain_pits_max_dist_XY
                if not b.any():
                    warn_pits.append(pit)
                    continue
                drain = drain[b]
                dxy = dxy[b]
            
            # calculate magnitudes
            s = (e[pit]-e[drain]) / dxy

            # connectivity info
            # TODO proportion calculation (_mk_connectivity_flats used elev?)
            pit_i += [pit for i in drain]
            pit_j += drain.tolist()
            pit_prop += s.tolist()
            
            # update pit magnitude and flats mask
            mag[ipit, jpit] = np.mean(s)
            flats[ipit, jpit] = False

        if warn_pits:
            warnings.warn("Warning %d pits had no place to drain to in this "
                          "chunk" % len(warn_pits))
        
        # Note: returning flats and mag here is not strictly necessary
        return (np.array(pit_i, 'int64'),
                np.array(pit_j, 'int64'),
                np.array(pit_prop, 'float64'),
                flats,
                mag)