def errReceived(self, data):
        """
        Connected process wrote to stderr
        """
        lines = data.splitlines()
        for line in lines:
            log_error("*** {name} stderr *** {line}", 
                    name=self.name,
                    line=self.errFilter(line))