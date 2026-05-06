def __has_edge(self, node1_name, node2_name, account_for_direction=True):
        """ Returns a boolean flag, telling if a tree has an edge with two nodes, specified by their names as arguments

        If a account_for_direction is specified as True, the order of specified node names has to relate to parent - child relation,
        otherwise both possibilities are checked
        """
        try:
            p1 = self.__get_node_by_name(name=node1_name)
            wdir = node2_name in (node.name for node in p1.children)
            if account_for_direction:
                return wdir
            else:
                p2 = self.__get_node_by_name(name=node2_name)
                return wdir or node1_name in (node.name for node in p2.children)
        except ValueError:
            return False