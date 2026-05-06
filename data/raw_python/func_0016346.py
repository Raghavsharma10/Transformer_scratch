def emailclients(self, tag=None, fromdate=None, todate=None):
        """
        Gets an overview of the email clients used to open your emails.
        This is only recorded when open tracking is enabled for that email.
        """
        return self.call("GET", "/stats/outbound/opens/emailclients", tag=tag, fromdate=fromdate, todate=todate)