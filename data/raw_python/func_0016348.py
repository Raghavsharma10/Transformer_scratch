def clicks(self, tag=None, fromdate=None, todate=None):
        """
        Gets total counts of unique links that were clicked.
        """
        return self.call("GET", "/stats/outbound/clicks", tag=tag, fromdate=fromdate, todate=todate)