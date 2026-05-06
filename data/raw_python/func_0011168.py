def is_forced_retry(self, method, status_code):
        """ Is this method/status code retryable? (Based on method/codes whitelists)
        """
        if self.method_whitelist and method.upper() not in self.method_whitelist:
            return False

        return self.status_forcelist and status_code in self.status_forcelist