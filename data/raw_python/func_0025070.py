def __get_deep_dependent_item_set(self, item, item_set) -> None:
        """Return the list of data items containing data that directly depends on data in this item."""
        if not item in item_set:
            item_set.add(item)
            with self.__dependency_tree_lock:
                for dependent in self.get_dependent_items(item):
                    self.__get_deep_dependent_item_set(dependent, item_set)