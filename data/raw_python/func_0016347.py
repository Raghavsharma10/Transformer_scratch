def readtimes(self, tag=None, fromdate=None, todate=None):
        """
        Gets the length of time that recipients read emails along with counts for each time.
        This is only recorded when open tracking is enabled for that email.
        Read time tracking stops at 20 seconds, so any read times above that will appear in the 20s+ field.
        """
        return self.call("GET", "/stats/outbound/opens/readtimes", tag=tag, fromdate=fromdate, todate=todate)