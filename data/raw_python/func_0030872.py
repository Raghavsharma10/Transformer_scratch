def build_fun(self, layer=None):
        """Calculate the merkle root and make references between nodes in the tree.
        Written in functional style purely for fun.
        """
        if not layer:
            if not self.leaves:
                raise MerkleError('The tree has no leaves and cannot be calculated.')
            layer = self.leaves[::]
        layer = self._build(layer)
        if len(layer) == 1:
            self.root = layer[0]
        else:
            self.build_fun(layer=layer)
        return self.root.val