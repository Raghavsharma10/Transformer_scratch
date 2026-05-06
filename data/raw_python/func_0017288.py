def newick(self, tree_format=0):
        "Returns newick represenation of the tree in its current state."
        # checks one of root's children for features and extra feats.
        if self.treenode.children:
            features = {"name", "dist", "support", "height", "idx"}
            testnode = self.treenode.children[0]
            extrafeat = {i for i in testnode.features if i not in features}
            features.update(extrafeat)
            return self.treenode.write(format=tree_format)