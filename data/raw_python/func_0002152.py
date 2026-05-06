def _select_date_range(self, lines):
        """Identify lines containing headers within the range begin_date to end_date.

        Parameters
        -----
        lines: list
            list of lines from the IGRA2 data file.

        """
        headers = []
        num_lev = []
        dates = []

        # Get indices of headers, and make a list of dates and num_lev
        for idx, line in enumerate(lines):
            if line[0] == '#':
                year, month, day, hour = map(int, line[13:26].split())

                # All soundings have YMD, most have hour
                try:
                    date = datetime.datetime(year, month, day, hour)
                except ValueError:
                    date = datetime.datetime(year, month, day)

                # Check date
                if self.begin_date <= date <= self.end_date:
                    headers.append(idx)
                    num_lev.append(int(line[32:36]))
                    dates.append(date)
                if date > self.end_date:
                    break

        if len(dates) == 0:
            # Break if no matched dates.
            # Could improve this later by showing the date range for the station.
            raise ValueError('No dates match selection.')

        # Compress body of data into a string
        begin_idx = min(headers)
        end_idx = max(headers) + num_lev[-1]

        # Make a boolean vector that selects only list indices within the time range
        selector = np.zeros(len(lines), dtype=bool)
        selector[begin_idx:end_idx + 1] = True
        selector[headers] = False
        body = ''.join([line for line in itertools.compress(lines, selector)])

        selector[begin_idx:end_idx + 1] = ~selector[begin_idx:end_idx + 1]
        header = ''.join([line for line in itertools.compress(lines, selector)])

        # expand date vector to match length of the body dataframe.
        dates_long = np.repeat(dates, num_lev)

        return body, header, dates_long, dates