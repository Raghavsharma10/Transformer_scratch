def pair(self, pair):
        '''Add a *pair* to the structure.'''
        if len(pair) == 1:
            # if only one value is passed, the value must implement a
            # score function which retrieve the first value of the pair
            # (score in zset, timevalue in timeseries, field value in
            # hashtable)
            return (pair[0].score(), pair[0])
        elif len(pair) != 2:
            raise TypeError('add expected 2 arguments, got {0}'
                            .format(len(pair)))
        else:
            return pair