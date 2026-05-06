def search(self):
        """Run the full search process.

        Simple public method to abstract the steps needed to produce a full
        search using the engine.
        """
        requests = self._format()
        serps = self._fetch(requests)
        urls = self._process(serps)
        details = self._fetch(urls)
        emails = self._extract()
        return {'emails': emails, 'processed': len(self.data)}