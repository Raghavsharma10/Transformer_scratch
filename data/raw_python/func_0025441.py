def insert_value(self, keys, value):
        """
            Insert a value (data item) into this tree node and then its
            children. This will be called in response to a new data item being
            inserted into the document. Also updates the tree node's cumulative
            child count.
        """
        self.count += 1
        if not self.key:
            self.__value_reverse_mapping[value] = keys
        if len(keys) == 0:
            self.values.append(value)
        else:
            key = keys[0]
            index = bisect.bisect_left(self.children, TreeNode(key, reversed=self.reversed))
            if index == len(self.children) or self.children[index].key != key:
                new_tree_node = TreeNode(key, list(), reversed=self.reversed)
                new_tree_node.child_inserted = self.child_inserted
                new_tree_node.child_removed = self.child_removed
                new_tree_node.tree_node_updated = self.tree_node_updated
                new_tree_node.__set_parent(self)
                self.children.insert(index, new_tree_node)
                if self.child_inserted:
                    self.child_inserted(self, index, new_tree_node)
            child = self.children[index]
            child.insert_value(keys[1:], value)
            if self.tree_node_updated:
                self.tree_node_updated(child)