def get_edge_list(self):
        """Get the list of Edge instances.

        This method returns the list of Edge instances
        composing the graph.
        """
        edge_objs = list()

        for obj_dict_list in self.obj_dict['edges'].values():
            edge_objs.extend([
                Edge(obj_dict=obj_d)
                for obj_d
                in obj_dict_list])

        return edge_objs