def remove_value(self, keys, value):
        """
            Remove a value (data item) from this tree node and its children.
            Also updates the tree node's cumulative child count.
        """
        self.count -= 1
        if not self.key:
            keys = self.__value_reverse_mapping[value]
            del self.__value_reverse_mapping[value]
        if len(keys) == 0:
            self.values.remove(value)
        else:
            key = keys[0]
            index = bisect.bisect_left(self.children, TreeNode(key, reversed=self.reversed))
            assert index != len(self.children) and self.children[index].key == key
            self.children[index].remove_value(keys[1:], value)
            if self.tree_node_updated:
                self.tree_node_updated(self.children[index])
            if self.children[index].count == 0:
                del self.children[index]
                if self.child_removed:
                    self.child_removed(self, index)