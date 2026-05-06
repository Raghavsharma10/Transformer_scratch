def safe_datetime_cast(self, col):
        """Parses string values into datetime.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.Series
        """
        casted_dates = pd.to_datetime(col[self.col_name], format=self.date_format, errors='coerce')

        if len(casted_dates[casted_dates.isnull()]):
            # This will raise an error for bad formatted data
            # but not for out of bonds or missing dates.
            slice_ = casted_dates.isnull() & ~col[self.col_name].isnull()
            col[slice_][self.col_name].apply(self.strptime_format)

        return casted_dates