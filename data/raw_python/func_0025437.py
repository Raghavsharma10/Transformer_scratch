def date_browser_selection_changed(self, selected_indexes):
        """
            Called to handle selection changes in the tree widget.

            This method should be connected to the on_selection_changed event. This method builds a list
            of keys represented by all selected items. It then provides date_filter to filter data items
            based on the list of keys. It then sets the filter into the document controller.

            :param selected_indexes: The selected indexes
            :type selected_indexes: list of ints
        """
        partial_date_filters = list()

        for index, parent_row, parent_id in selected_indexes:
            item_model_controller = self.item_model_controller
            tree_node = item_model_controller.item_value("tree_node", index, parent_id)
            partial_date_filters.append(ListModel.PartialDateFilter("created_local", *tree_node.keys))

        if len(partial_date_filters) > 0:
            self.__date_filter = ListModel.OrFilter(partial_date_filters)
        else:
            self.__date_filter = None

        self.__update_filter()