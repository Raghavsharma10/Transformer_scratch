def snapshot(self):
        """Return a new library item which is a copy of this one with any dynamic behavior made static."""
        data_item = self.__class__()
        # data format (temporary until moved to buffered data source)
        data_item.large_format = self.large_format
        data_item.set_data_and_metadata(copy.deepcopy(self.data_and_metadata), self.data_modified)
        # metadata
        data_item.created = self.created
        data_item.timezone = self.timezone
        data_item.timezone_offset = self.timezone_offset
        data_item.metadata = self.metadata
        data_item.title = self.title
        data_item.caption = self.caption
        data_item.description = self.description
        data_item.session_id = self.session_id
        data_item.session_data = copy.deepcopy(self.session_data)
        return data_item