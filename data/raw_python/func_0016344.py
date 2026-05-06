def opens(self, tag=None, fromdate=None, todate=None):
        """
        Gets total counts of recipients who opened your emails.
        This is only recorded when open tracking is enabled for that email.
        """
        return self.call("GET", "/stats/outbound/opens", tag=tag, fromdate=fromdate, todate=todate)