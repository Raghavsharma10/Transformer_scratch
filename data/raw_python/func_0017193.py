def write(self, 
        features=None, 
        outfile=None, 
        format=0, 
        is_leaf_fn=None,
        format_root_node=False, 
        dist_formatter=None, 
        support_formatter=None,
        name_formatter=None):

        """
        Returns the newick representation of current node. Several
        arguments control the way in which extra data is shown for
        every node:

        Parameters:
        -----------
        features: 
            a list of feature names to be exported using the Extended Newick 
            Format (i.e. features=["name", "dist"]). Use an empty list to 
            export all available features in each node (features=[])

        outfile:
            writes the output to a given file

        format: 
            defines the newick standard used to encode the tree. 

        format_root_node: 
            If True, it allows features and branch information from root node
            to be exported as a part of the newick text string. For newick 
            compatibility reasons, this is False by default.

        is_leaf_fn: 
            See :func:`TreeNode.traverse` for documentation.

        **Example:**
             t.get_newick(features=["species","name"], format=1)
        """
        nw = write_newick(self, features=features,
                          format=format,
                          is_leaf_fn=is_leaf_fn,
                          format_root_node=format_root_node,
                          dist_formatter=dist_formatter,
                          support_formatter=support_formatter,
                          name_formatter=name_formatter)

        if outfile is not None:
            with open(outfile, "w") as OUT:
                OUT.write(nw)
        else:
            return nw