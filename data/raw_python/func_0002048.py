def get_scan_log_lines(self, source_id, scan_id):
        """
        Get the log text for a Scan

        :rtype: Iterator over log lines.
        """
        return self.client.get_manager(Scan).get_log_lines(source_id=source_id, scan_id=scan_id)