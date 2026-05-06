def update_idxs(self):
        "set root idx highest, tip idxs lowest ordered as ladderized"
        # internal nodes: root is highest idx
        idx = self.ttree.nnodes - 1
        for node in self.ttree.treenode.traverse("levelorder"):
            if not node.is_leaf():
                node.add_feature("idx", idx)
                if not node.name:
                    node.name = str(idx)
                idx -= 1

        # external nodes: lowest numbers are for tips (0-N)
        for node in self.ttree.treenode.get_leaves():
            node.add_feature("idx", idx)
            if not node.name:
                node.name = str(idx)
            idx -= 1