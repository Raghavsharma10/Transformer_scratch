def filter(self, filter_func, reverse=False):
        """Filter current log lines by a given filter function.

        This allows to drill down data out of the log file by filtering the
        relevant log lines to analyze.

        For example, filter by a given IP so only log lines for that IP are
        further processed with commands (top paths, http status counter...).

        :param filter_func: [required] Filter method, see filters.py for all
          available filters.
        :type filter_func: function
        :param reverse: negate the filter (so accept all log lines that return
          ``False``).
        :type reverse: boolean
        :returns: a new instance of Log containing only log lines
          that passed the filter function.
        :rtype: :class:`Log`
        """
        new_log_file = Log()
        new_log_file.logfile = self.logfile

        new_log_file.total_lines = 0

        new_log_file._valid_lines = []
        new_log_file._invalid_lines = self._invalid_lines[:]

        # add the reverse conditional outside the loop to keep the loop as
        # straightforward as possible
        if not reverse:
            for i in self._valid_lines:
                if filter_func(i):
                    new_log_file.total_lines += 1
                    new_log_file._valid_lines.append(i)
        else:
            for i in self._valid_lines:
                if not filter_func(i):
                    new_log_file.total_lines += 1
                    new_log_file._valid_lines.append(i)

        return new_log_file