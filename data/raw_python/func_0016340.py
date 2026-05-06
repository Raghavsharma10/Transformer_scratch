def sends(self, tag=None, fromdate=None, todate=None):
        """
        Gets a total count of emails you’ve sent out.
        """
        return self.call("GET", "/stats/outbound/sends", tag=tag, fromdate=fromdate, todate=todate)