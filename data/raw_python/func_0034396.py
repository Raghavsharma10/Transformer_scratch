def __get_node_by_name(self, name):
        """ Returns a first TreeNode object, which name matches the specified argument

        :raises: ValueError (if no node with specified name is present in the tree)
        """
        try:
            for entry in filter(lambda x: x.name == name, self.nodes()):
                return entry
        except StopIteration:
            raise ValueError("Attempted to retrieve a non-existing tree node with name: {name}"
                             "".format(name=name))