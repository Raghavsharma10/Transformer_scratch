def get_distance(self, node1_name, node2_name):
        """ Returns a length of an edge / path, if exists, from the current tree

        :param node1_name: a first node name in current tree
        :param node2_name: a second node name in current tree
        :return: a length of specified by a pair of vertices edge / path
        :rtype: `Number`
        :raises: ValueError, if requested a length of an edge, that is not present in current tree
        """
        return self.__root.get_distance(target=node1_name, target2=node2_name)