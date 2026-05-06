def node_path_transforms(self, node):
        """Return the list of transforms along the path to another node.
        
        The transforms are listed in reverse order, such that the last 
        transform should be applied first when mapping from this node to 
        the other.

        Parameters
        ----------
        node : instance of Node
            The other node.

        Returns
        -------
        transforms : list
            A list of Transform instances.
        """
        a, b = self.node_path(node)
        return ([n.transform for n in a[:-1]] + 
                [n.transform.inverse for n in b])[::-1]