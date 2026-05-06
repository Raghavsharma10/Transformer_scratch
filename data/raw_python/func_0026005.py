def hasExpired(self):
        """
        :return: true if the lastUpdateTime is more than maxAge seconds ago.
        """
        return (datetime.datetime.utcnow() - self.lastUpdateTime).total_seconds() > self.maxAgeSeconds