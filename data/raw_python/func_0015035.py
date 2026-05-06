def cmd_requests_per_minute(self):
        """Generates statistics on how many requests were made per minute.

        .. note::
          Try to combine it with time constrains (``-s`` and ``-d``) as this
          command output can be huge otherwise.
        """
        if len(self._valid_lines) == 0:
            return

        current_minute = self._valid_lines[0].accept_date
        current_minute_counter = 0
        requests = []
        one_minute = timedelta(minutes=1)

        def format_and_append(append_to, date, counter):
            seconds_and_micro = timedelta(
                seconds=date.second,
                microseconds=date.microsecond,
            )
            minute_formatted = date - seconds_and_micro
            append_to.append((minute_formatted, counter))

        # note that _valid_lines is kept sorted by date
        for line in self._valid_lines:
            line_date = line.accept_date
            if line_date - current_minute < one_minute and \
                    line_date.minute == current_minute.minute:
                current_minute_counter += 1

            else:
                format_and_append(
                    requests,
                    current_minute,
                    current_minute_counter,
                )
                current_minute_counter = 1
                current_minute = line_date

        if current_minute_counter > 0:
            format_and_append(
                requests,
                current_minute,
                current_minute_counter,
            )

        return requests