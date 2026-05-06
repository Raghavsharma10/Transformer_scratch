def bounces(self, tag=None, fromdate=None, todate=None):
        """
        Gets total counts of emails you’ve sent out that have been returned as bounced.
        """
        return self.call("GET", "/stats/outbound/bounces", tag=tag, fromdate=fromdate, todate=todate)