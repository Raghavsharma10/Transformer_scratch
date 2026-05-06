def append(self, node_name, tree, copy=False):
        """ Append a specified tree (represented by a root TreeNode element) to the node, specified by its name

        :param copy: a flag denoting if the appended tree has to be added as is, or is the deepcopy of it has to be used
        :type copy: Boolean
        :raises: ValueError (if no node with a specified name, to which the specified tree has to be appended, is present in the current tree)
        """
        self.multicolors_are_up_to_date = False
        tree_to_append = tree.__root if not copy else deepcopy(tree.__root)
        self.__get_node_by_name(node_name).add_child(tree_to_append)