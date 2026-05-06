def tracked(self, tag=None, fromdate=None, todate=None):
        """
        Gets a total count of emails you’ve sent with open tracking or link tracking enabled.
        """
        return self.call("GET", "/stats/outbound/tracked", tag=tag, fromdate=fromdate, todate=todate)