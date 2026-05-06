def overview(self, tag=None, fromdate=None, todate=None):
        """
        Gets a brief overview of statistics for all of your outbound email.
        """
        return self.call("GET", "/stats/outbound", tag=tag, fromdate=fromdate, todate=todate)