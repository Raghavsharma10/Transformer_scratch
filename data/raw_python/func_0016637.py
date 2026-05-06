def _grow_trees(self):
        """
        Adds new trees to the forest according to the specified growth method.
        """
        if self.grow_method == GROW_AUTO_INCREMENTAL:
            self.tree_kwargs['auto_grow'] = True
        
        while len(self.trees) < self.size:
            self.trees.append(Tree(data=self.data, **self.tree_kwargs))