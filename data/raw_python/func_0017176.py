def add_child(self, child=None, name=None, dist=None, support=None):
        """
        Adds a new child to this node. If child node is not suplied
        as an argument, a new node instance will be created.
        
        Parameters
        ----------

        child: 
            the node instance to be added as a child.
        name: 
            the name that will be given to the child.
        dist: 
            the distance from the node to the child.
        support': 
            the support value of child partition.

        Returns:
        --------
            The child node instance
        """

        if child is None:
            child = self.__class__()
        if name is not None:
            child.name = name
        if dist is not None:
            child.dist = dist
        if support is not None:
            child.support = support

        self.children.append(child)
        child.up = self
        return child