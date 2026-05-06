def insert_data_item(self, before_index, data_item, auto_display: bool = True) -> None:
        """Insert a new data item into document model.

        This method is NOT threadsafe.
        """
        assert data_item is not None
        assert data_item not in self.data_items
        assert before_index <= len(self.data_items) and before_index >= 0
        assert data_item.uuid not in self.__uuid_to_data_item
        # update the session
        data_item.session_id = self.session_id
        # insert in internal list
        self.__insert_data_item(before_index, data_item, do_write=True)
        # automatically add a display
        if auto_display:
            display_item = DisplayItem.DisplayItem(data_item=data_item)
            self.append_display_item(display_item)