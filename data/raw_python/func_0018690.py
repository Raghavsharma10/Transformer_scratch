def _prepare_window(self, start_time: datetime) -> None:
        """
        Prepares window if any is specified.
        :param start_time: The anchor block start_time from where the window
        should be generated.
        """
        # evaluate window first which sets the correct window in the store
        store = self._schema.schema_loader.get_store(
            self._schema.source.store_schema.fully_qualified_name)
        if Type.is_type_equal(self._schema.window_type, Type.DAY) or Type.is_type_equal(
                self._schema.window_type, Type.HOUR):
            block_list = self._load_blocks(
                store.get_range(
                    Key(self._schema.source.key_type, self._identity, self._schema.source.name),
                    start_time, self._get_end_time(start_time)))
        else:
            block_list = self._load_blocks(
                store.get_range(
                    Key(self._schema.source.key_type, self._identity, self._schema.source.name),
                    start_time, None, self._schema.window_value))

        self._window_source = _WindowSource(block_list)
        self._validate_view()