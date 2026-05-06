def get_edge_init_data(self, fn, save_path=None):
        """
        Creates the initialization data from the edge structure
        """

        edge_init_data = {key: self.edges[fn][key].get('data') for key in
                          self.edges[fn].keys()}
        edge_init_done = {key: self.edges[fn][key].get('done') for key in
                          self.edges[fn].keys()}
        edge_init_todo = {key: self.edges[fn][key].get('todo') for key in
                          self.edges[fn].keys()}
        return edge_init_data, edge_init_done, edge_init_todo