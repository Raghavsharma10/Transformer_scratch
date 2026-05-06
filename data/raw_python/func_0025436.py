def update_all_nodes(self):
        """ Update all tree item displays if needed. Usually for count updates. """
        item_model_controller = self.item_model_controller
        if item_model_controller:
            if self.__node_counts_dirty:
                for item in self.__mapping.values():
                    if "tree_node" in item.data:  # don't update the root node
                        tree_node = item.data["tree_node"]
                        item.data["display"] = self.__display_for_tree_node(tree_node)
                        item_model_controller.data_changed(item.row, item.parent.row, item.parent.id)
                self.__node_counts_dirty = False