def set_all_neighbors_data(self, data, done,  key):
        """
        Given they 'key' tile's data, assigns this information to all
        neighboring tiles
        """
        # The order of this for loop is important because the topleft gets
        # it's data from the left neighbor, which should have already been
        # updated...
        for side in ['left', 'right', 'top', 'bottom', 'topleft',
                     'topright', 'bottomleft', 'bottomright']:
            self.set_neighbor_data(side, data, key, 'data')
#            self.set_neighbor_data(side, todo, key, 'todo')
            self.set_neighbor_data(side, done, key, 'done')