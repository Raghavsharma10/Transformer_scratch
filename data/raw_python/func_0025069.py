def get_dependent_items(self, item) -> typing.List:
        """Return the list of data items containing data that directly depends on data in this item."""
        with self.__dependency_tree_lock:
            return copy.copy(self.__dependency_tree_source_to_target_map.get(weakref.ref(item), list()))