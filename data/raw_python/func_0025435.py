def __remove_child(self, parent_tree_node, index):
        """
            Called from the root tree node when a node is removed from the tree. This method removes it into the
            item model controller.
        """
        # get parent and item
        parent_item = self.__mapping[id(parent_tree_node)]
        # manage the item model
        self.item_model_controller.begin_remove(index, index, parent_item.row, parent_item.id)
        child_item = parent_item.children[index]
        parent_item.remove_child(child_item)
        self.__mapping.pop(id(child_item.data["tree_node"]))
        self.item_model_controller.end_remove()