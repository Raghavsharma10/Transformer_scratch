def to_datetime(self, column):
        '''
        This function converts epoch timestamps to datetimes.

        :param column: column to convert from current state -> datetime
        '''
        if column in self:
            if self[column].dtype in NUMPY_NUMERICAL:
                self[column] = pd.to_datetime(self[column], unit='s')
            else:
                self[column] = pd.to_datetime(self[column], utc=True)