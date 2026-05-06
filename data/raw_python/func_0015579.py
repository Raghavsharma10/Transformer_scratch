def reset(self):
        '''Reset the histogram to a pristine state
        '''
        for index in range(self.counts_len):
            self.counts[index] = 0
        self.total_count = 0
        self.min_value = sys.maxsize
        self.max_value = 0
        self.start_time_stamp_msec = sys.maxsize
        self.end_time_stamp_msec = 0