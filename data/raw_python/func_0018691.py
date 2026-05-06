def _get_end_time(self, start_time: datetime) -> datetime:
        """
        Generates the end time to be used for the store range query.
        :param start_time: Start time to use as an offset to calculate the end time
        based on the window type in the schema.
        :return:
        """
        if Type.is_type_equal(self._schema.window_type, Type.DAY):
            return start_time + timedelta(days=self._schema.window_value)
        elif Type.is_type_equal(self._schema.window_type, Type.HOUR):
            return start_time + timedelta(hours=self._schema.window_value)