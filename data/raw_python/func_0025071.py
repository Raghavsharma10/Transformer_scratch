def get_dependent_data_items(self, data_item: DataItem.DataItem) -> typing.List[DataItem.DataItem]:
        """Return the list of data items containing data that directly depends on data in this item."""
        with self.__dependency_tree_lock:
            return [data_item for data_item in self.__dependency_tree_source_to_target_map.get(weakref.ref(data_item), list()) if isinstance(data_item, DataItem.DataItem)]