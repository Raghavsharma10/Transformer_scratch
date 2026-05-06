def _download(self):
        """Download the page."""
        print("Downloading!")
        def parse(result):
            print("Got %r back from Yahoo." % (result,))
            values = result.strip().split(",")
            self._value = float(values[1])
        d = getPage(
            "http://download.finance.yahoo.com/d/quotes.csv?e=.csv&f=c4l1&s=%s=X"
            % (self._name,))
        d.addCallback(parse)
        d.addErrback(log.err)
        return d