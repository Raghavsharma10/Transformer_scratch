def scatterplot(self, cycle_data):
        """Return scatterplot data for the cycle times in `cycle_data`. Returns
        a data frame containing only those items in `cycle_data` where values
        are set for `completed_timestamp` and `cycle_time`, and with those two
        columns as the first two, both normalised to whole days, and with
        `completed_timestamp` renamed to `completed_date`.
        """

        columns = list(cycle_data.columns)
        columns.remove('cycle_time')
        columns.remove('completed_timestamp')
        columns = ['completed_timestamp', 'cycle_time'] + columns

        data = (
            cycle_data[columns]
            .dropna(subset=['cycle_time', 'completed_timestamp'])
            .rename(columns={'completed_timestamp': 'completed_date'})
        )

        data['cycle_time'] = data['cycle_time'].astype('timedelta64[D]')
        data['completed_date'] = data['completed_date'].map(pd.Timestamp.date)

        return data