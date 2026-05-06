def output_interval_histogram(self,
                                  histogram,
                                  start_time_stamp_sec=0,
                                  end_time_stamp_sec=0,
                                  max_value_unit_ratio=1000000.0):
        '''Output an interval histogram, with the given timestamp and a
        configurable maxValueUnitRatio.
        (note that the specified timestamp will be used, and the timestamp in
        the actual histogram will be ignored).
        The max value reported with the interval line will be scaled by the
        given max_value_unit_ratio.
        The histogram start and end timestamps are assumed to be in msec units.
        Logging will be in seconds, realtive by a base time
        The default base time is 0.

        By covention, histogram start/end time are generally stamped with
        absolute times in msec since the epoch. For logging with absolute time
        stamps, the base time would remain zero. For
        logging with relative time stamps (time since a start point),
        Params:
            histogram The interval histogram to log.
            start_time_stamp_sec The start timestamp to log with the
                interval histogram, in seconds.
                default: using the start/end timestamp indicated in the histogram
            end_time_stamp_sec The end timestamp to log with the interval
                histogram, in seconds.
                default: using the start/end timestamp indicated in the histogram
            max_value_unit_ratio The ratio by which to divide the histogram's max
                value when reporting on it.
                default: 1,000,000 (which is the msec : nsec ratio
        '''
        if not start_time_stamp_sec:
            start_time_stamp_sec = \
                (histogram.get_start_time_stamp() - self.base_time) / 1000.0
        if not end_time_stamp_sec:
            end_time_stamp_sec = (histogram.get_end_time_stamp() - self.base_time) / 1000.0
        cpayload = histogram.encode()
        self.log.write("%f,%f,%f,%s\n" %
                       (start_time_stamp_sec,
                        end_time_stamp_sec - start_time_stamp_sec,
                        histogram.get_max_value() // max_value_unit_ratio,
                        cpayload.decode('utf-8')))