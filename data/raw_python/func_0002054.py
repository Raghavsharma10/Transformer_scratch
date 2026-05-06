def get_log_lines(self):
        """
        Get the log text for a scan object

        :rtype: Iterator over log lines.
        """
        rel = self._client.reverse_url('SCAN', self.url)
        return self._manager.get_log_lines(**rel)