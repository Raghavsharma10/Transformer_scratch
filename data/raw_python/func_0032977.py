def requires(self):
        """ Index all pages. """
        for url in NEWSPAPERS:
            yield IndexPage(url=url, date=self.date)