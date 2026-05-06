def _decode_next_interval_histogram(self,
                                        dest_histogram,
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
            dest_histogram if None, created a new histogram, else adds
                           the new interval histogram to it
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
        while 1:
            line = self.input_file.readline()
            if not line:
                return None
            if line[0] == '#':
                match_res = re_start_time.match(line)
                if match_res:
                    self.start_time_sec = float(match_res.group(1))
                    self.observed_start_time = True
                    continue
                match_res = re_base_time.match(line)
                if match_res:
                    self.base_time_sec = float(match_res.group(1))
                    self.observed_base_time = True
                    continue

            match_res = re_histogram_interval.match(line)
            if not match_res:
                # probably a legend line that starts with "\"StartTimestamp"
                continue
            # Decode: startTimestamp, intervalLength, maxTime, histogramPayload
            # Timestamp is expected to be in seconds
            log_time_stamp_in_sec = float(match_res.group(1))
            interval_length_sec = float(match_res.group(2))
            cpayload = match_res.group(4)

            if not self.observed_start_time:
                # No explicit start time noted. Use 1st observed time:
                self.start_time_sec = log_time_stamp_in_sec
                self.observed_start_time = True

            if not self.observed_base_time:
                # No explicit base time noted.
                # Deduce from 1st observed time (compared to start time):
                if log_time_stamp_in_sec < self.start_time_sec - (365 * 24 * 3600.0):
                    # Criteria Note: if log timestamp is more than a year in
                    # the past (compared to StartTime),
                    # we assume that timestamps in the log are not absolute
                    self.base_time_sec = self.start_time_sec
                else:
                    # Timestamps are absolute
                    self.base_time_sec = 0.0
                self.observed_base_time = True

            absolute_start_time_stamp_sec = \
                log_time_stamp_in_sec + self.base_time_sec
            offset_start_time_stamp_sec = \
                absolute_start_time_stamp_sec - self.start_time_sec

            # Timestamp length is expect to be in seconds
            absolute_end_time_stamp_sec = \
                absolute_start_time_stamp_sec + interval_length_sec

            if absolute:
                start_time_stamp_to_check_range_on = absolute_start_time_stamp_sec
            else:
                start_time_stamp_to_check_range_on = offset_start_time_stamp_sec

            if start_time_stamp_to_check_range_on < range_start_time_sec:
                continue

            if start_time_stamp_to_check_range_on > range_end_time_sec:
                return None
            if dest_histogram:
                # add the interval histogram to the destination histogram
                histogram = dest_histogram
                histogram.decode_and_add(cpayload)
            else:
                histogram = HdrHistogram.decode(cpayload)
                histogram.set_start_time_stamp(absolute_start_time_stamp_sec * 1000.0)
                histogram.set_end_time_stamp(absolute_end_time_stamp_sec * 1000.0)
            return histogram