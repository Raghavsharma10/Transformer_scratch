def sort_descendants(self, attr="name"):
        """
        This function sort the branches of a given tree by
        considerening node names. After the tree is sorted, nodes are
        labeled using ascendent numbers.  This can be used to ensure
        that nodes in a tree with the same node names are always
        labeled in the same way. Note that if duplicated names are
        present, extra criteria should be added to sort nodes.

        Unique id is stored as a node._nid attribute
        """
        node2content = self.get_cached_content(store_attr=attr, container_type=list)
        for n in self.traverse():
            if not n.is_leaf():
                n.children.sort(key=lambda x: str(sorted(node2content[x])))