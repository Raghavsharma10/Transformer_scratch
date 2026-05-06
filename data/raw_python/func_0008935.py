def update_edges(self, elev_fn, dem_proc):
        """
        After finishing a calculation, this will update the neighbors and the
        todo for that tile
        """
        interp = self.build_interpolator(dem_proc)
        self.update_edge_todo(elev_fn, dem_proc)
        self.set_neighbor_data(elev_fn, dem_proc, interp)