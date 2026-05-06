def throughput_data(self, cycle_data, frequency='1D',pointscolumn= None):
        """Return a data frame with columns `completed_timestamp` of the
        given frequency, either
        `count`, where count is the number of items
        'sum', where sum is the sum of value specified by pointscolumn. Expected to be 'StoryPoints'
        completed at that timestamp (e.g. daily).
        """
        if len(cycle_data)<1:
           return None # Note completed items yet, return None

        if pointscolumn:
            return cycle_data[['completed_timestamp', pointscolumn]] \
                .rename(columns={pointscolumn: 'sum'}) \
                .groupby('completed_timestamp').sum() \
                .resample(frequency).sum() \
                .fillna(0)
        else:
            return cycle_data[['completed_timestamp', 'key']] \
                .rename(columns={'key': 'count'}) \
                .groupby('completed_timestamp').count() \
                .resample(frequency).sum() \
                .fillna(0)