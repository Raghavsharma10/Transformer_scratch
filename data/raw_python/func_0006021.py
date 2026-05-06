def get_browser_log(self, levels=None):
        """Gets the console log of the browser

        @type levels:
        @return: List of browser log entries
        """
        logs = self.driver.get_log('browser')
        self.browser_logs += logs
        if levels is not None:
            logs = [entry for entry in logs if entry.get(u'level') in levels]
        return logs