def spam(self, tag=None, fromdate=None, todate=None):
        """
        Gets a total count of recipients who have marked your email as spam.
        """
        return self.call("GET", "/stats/outbound/spam", tag=tag, fromdate=fromdate, todate=todate)