def outLineReceived(self, line):
        """
        Handle data via stdout linewise. This is useful if you turned off
        buffering.

        In your subclass, override this if you want to handle the line as a
        protocol line in addition to logging it. (You may upcall this function
        safely.)
        """
        log_debug('<<< {name} stdout >>> {line}', 
                name=self.name,
                line=self.outFilter(line))