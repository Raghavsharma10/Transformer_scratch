def clicks_platforms(self, tag=None, fromdate=None, todate=None):
        """
        Gets an overview of the browser platforms used to open your emails.
        This is only recorded when Link Tracking is enabled for that email.
        """
        return self.call("GET", "/stats/outbound/clicks/platforms", tag=tag, fromdate=fromdate, todate=todate)