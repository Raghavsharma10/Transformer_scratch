def get_node_dict(self, return_internal=False, return_nodes=False):
        """
        Return node labels as a dictionary mapping {idx: name} where idx is 
        the order of nodes in 'preorder' traversal. Used internally by the
        func .get_node_values() to return values in proper order. 

        return_internal: if True all nodes are returned, if False only tips.
        return_nodes: if True returns TreeNodes, if False return node names.
        """
        if return_internal:
            if return_nodes:
                return {
                    i.idx: i for i in self.treenode.traverse("preorder")
                }
            else:
                return {
                    i.idx: i.name for i in self.treenode.traverse("preorder")
                }
        else:
            if return_nodes:
                return {
                    i.idx: i for i in self.treenode.traverse("preorder")
                    if i.is_leaf()
                }
            else:
                return {
                    i.idx: i.name for i in self.treenode.traverse("preorder")
                    if i.is_leaf()
                }