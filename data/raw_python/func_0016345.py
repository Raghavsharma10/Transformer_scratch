def opens_platforms(self, tag=None, fromdate=None, todate=None):
        """
        Gets an overview of the platforms used to open your emails.
        This is only recorded when open tracking is enabled for that email.
        """
        return self.call("GET", "/stats/outbound/opens/platforms", tag=tag, fromdate=fromdate, todate=todate)