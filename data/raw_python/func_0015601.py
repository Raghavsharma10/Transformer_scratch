def get_next_interval_histogram(self,
                                    range_start_time_sec=0.0,
                                    range_end_time_sec=sys.maxsize,
                                    absolute=False):
        '''Read the next interval histogram from the log, if interval falls
        within an absolute or relative time range.

        Timestamps are assumed to appear in order in the log file, and as such
        this method will return a null upon encountering a timestamp larger than
        range_end_time_sec.

        Relative time range:
            the range is assumed to be in seconds relative to
            the actual timestamp value found in each interval line in the log
        Absolute time range:
            Absolute timestamps are calculated by adding the timestamp found
            with the recorded interval to the [latest, optional] start time
            found in the log. The start time is indicated in the log with
            a "#[StartTime: " followed by the start time in seconds.

        Params:

            range_start_time_sec The absolute or relative start of the expected
                                 time range, in seconds.
            range_start_time_sec The absolute or relative end of the expected
                                  time range, in seconds.
            absolute Defines if the passed range is absolute or relative

        Return:
            Returns an histogram object if an interval line was found with an
            associated start timestamp value that falls between start_time_sec and
            end_time_sec,
            or null if no such interval line is found.
            Upon encountering any unexpected format errors in reading the next
            interval from the file, this method will return None.

            The histogram returned will have it's timestamp set to the absolute
            timestamp calculated from adding the interval's indicated timestamp
            value to the latest [optional] start time found in the log.

        Exceptions:
            ValueError if there is a syntax error in one of the float fields
        '''
        return self._decode_next_interval_histogram(None,
                                                    range_start_time_sec,
                                                    range_end_time_sec,
                                                    absolute)