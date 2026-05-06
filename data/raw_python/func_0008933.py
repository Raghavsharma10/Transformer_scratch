def set_neighbor_data(self, elev_fn, dem_proc, interp=None):
        """
        From the elevation filename, we can figure out and load the data and
        done arrays.
        """
        if interp is None:
            interp = self.build_interpolator(dem_proc)
        opp = {'top': 'bottom', 'left': 'right'}
        for key in self.neighbors[elev_fn].keys():
            tile = self.neighbors[elev_fn][key]
            if tile == '':
                continue
            oppkey = key
            for me, neigh in opp.iteritems():
                if me in key:
                    oppkey = oppkey.replace(me, neigh)
                else:
                    oppkey = oppkey.replace(neigh, me)
            opp_edge = self.neighbors[tile][oppkey]
            if opp_edge == '':
                continue

            interp.values = dem_proc.uca[::-1, :]
#            interp.values[:, 0] = np.ravel(dem_proc.uca)  # for other interp.
            # for the top-left tile we have to set the bottom and right edges
            # of that tile, so two edges for those tiles
            for key_ed in oppkey.split('-'):
                self.edges[tile][key_ed].set_data('data', interp)

            interp.values = dem_proc.edge_done[::-1, :].astype(float)
#            interp.values[:, 0] = np.ravel(dem_proc.edge_done)
            for key_ed in oppkey.split('-'):
                self.edges[tile][key_ed].set_data('done', interp)