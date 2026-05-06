def getCheck(self, checkid):
        """Returns a detailed description of a specified check."""

        check = PingdomCheck(self, {'id': checkid})
        check.getDetails()
        return check