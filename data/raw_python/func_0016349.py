def browserfamilies(self, tag=None, fromdate=None, todate=None):
        """
        Gets an overview of the browsers used to open links in your emails.
        This is only recorded when Link Tracking is enabled for that email.
        """
        return self.call("GET", "/stats/outbound/clicks/browserfamilies", tag=tag, fromdate=fromdate, todate=todate)