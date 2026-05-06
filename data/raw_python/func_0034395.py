def add_edge(self, node1_name, node2_name, edge_length=DEFAULT_EDGE_LENGTH):
        """ Adds a new edge to the current tree with specified characteristics

        Forbids addition of an edge, if a parent node is not present
        Forbids addition of an edge, if a child node already exists

        :param node1_name: name of the parent node, to which an edge shall be added
        :param node2_name: name of newly added child node
        :param edge_length: a length of specified edge
        :return: nothing, inplace changes
        :raises: ValueError (if parent node IS NOT present in the tree, or child node IS already present in the tree)
        """
        if not self.__has_node(name=node1_name):
            raise ValueError("Can not add an edge to a non-existing node {name}".format(name=node1_name))
        if self.__has_node(name=node2_name):
            raise ValueError("Can not add an edge to already existing node {name}".format(name=node2_name))
        self.multicolors_are_up_to_date = False
        self.__get_node_by_name(name=node1_name).add_child(name=node2_name, dist=edge_length)