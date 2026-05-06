def traverse(self, strategy="levelorder", is_leaf_fn=None):
        """ Returns an iterator to traverse tree under this node.
        
        Parameters:
        -----------
        strategy: 
            set the way in which tree will be traversed. Possible 
            values are: "preorder" (first parent and then children)
            'postorder' (first children and the parent) and 
            "levelorder" (nodes are visited in order from root to leaves)

        is_leaf_fn: 
            If supplied, ``is_leaf_fn`` function will be used to 
            interrogate nodes about if they are terminal or internal. 
            ``is_leaf_fn`` function should receive a node instance as first
            argument and return True or False. Use this argument to 
            traverse a tree by dynamically collapsing internal nodes matching
            ``is_leaf_fn``.
        """
        if strategy == "preorder":
            return self._iter_descendants_preorder(is_leaf_fn=is_leaf_fn)
        elif strategy == "levelorder":
            return self._iter_descendants_levelorder(is_leaf_fn=is_leaf_fn)
        elif strategy == "postorder":
            return self._iter_descendants_postorder(is_leaf_fn=is_leaf_fn)