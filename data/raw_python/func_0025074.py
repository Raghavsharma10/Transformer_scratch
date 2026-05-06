def begin_data_item_live(self, data_item):
        """Begins a live state for the data item.

        The live state is propagated to dependent data items.

        This method is thread safe. See slow_test_dependent_data_item_removed_while_live_data_item_becomes_unlive.
        """
        with self.__live_data_items_lock:
            old_live_count = self.__live_data_items.get(data_item.uuid, 0)
            self.__live_data_items[data_item.uuid] = old_live_count + 1
        if old_live_count == 0:
            data_item._enter_live_state()
            for dependent_data_item in self.get_dependent_data_items(data_item):
                self.begin_data_item_live(dependent_data_item)