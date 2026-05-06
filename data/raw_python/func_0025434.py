def __insert_child(self, parent_tree_node, index, tree_node):
        """
            Called from the root tree node when a new node is inserted into tree. This method creates properties
            to represent the node for display and inserts it into the item model controller.
        """
        # manage the item model
        parent_item = self.__mapping[id(parent_tree_node)]
        self.item_model_controller.begin_insert(index, index, parent_item.row, parent_item.id)
        properties = {
            "display": self.__display_for_tree_node(tree_node),
            "tree_node": tree_node  # used for removal and other lookup
        }
        item = self.item_model_controller.create_item(properties)
        parent_item.insert_child(index, item)
        self.__mapping[id(tree_node)] = item
        self.item_model_controller.end_insert()