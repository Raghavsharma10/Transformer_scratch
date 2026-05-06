def update_edge_todo(self, elev_fn, dem_proc):
        """
        Can figure out how to update the todo based on the elev filename
        """
        for key in self.edges[elev_fn].keys():
            self.edges[elev_fn][key].set_data('todo', data=dem_proc.edge_todo)