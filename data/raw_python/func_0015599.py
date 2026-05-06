def output_start_time(self, start_time_msec):
        '''Log a start time in the log.
        Params:
            start_time_msec time (in milliseconds) since the absolute start time (the epoch)
        '''
        self.log.write("#[StartTime: %f (seconds since epoch), %s]\n" %
                       (float(start_time_msec) / 1000.0,
                        datetime.fromtimestamp(start_time_msec).iso_format(' ')))