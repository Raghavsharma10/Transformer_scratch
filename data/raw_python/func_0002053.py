def get_log_lines(self, source_id, scan_id):
        """
        Get the log text for a scan object
        :rtype: Iterator over log lines.
        """
        target_url = self.client.get_url('SCAN', 'GET', 'log', {'source_id': source_id, 'scan_id': scan_id})
        r = self.client.request('GET', target_url, headers={'Accept': 'text/plain'}, stream=True)
        return r.iter_lines(decode_unicode=True)