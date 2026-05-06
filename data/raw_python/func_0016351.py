def location(self, tag=None, fromdate=None, todate=None):
        """
        Gets an overview of which part of the email links were clicked from (HTML or Text).
        This is only recorded when Link Tracking is enabled for that email.
        """
        return self.call("GET", "/stats/outbound/clicks/location", tag=tag, fromdate=fromdate, todate=todate)