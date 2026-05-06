def feed(self, data):
        """
        Main method for purifying HTML (overrided)
        """
        self.reset_purified()
        HTMLParser.feed(self, data)
        return self.html()