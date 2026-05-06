def get_node_list(self):
        """Get the list of Node instances.

        This method returns the list of Node instances
        composing the graph.
        """
        node_objs = list()

        for obj_dict_list in self.obj_dict['nodes'].values():
            node_objs.extend([
                Node(obj_dict=obj_d)
                for obj_d
                in obj_dict_list])

        return node_objs