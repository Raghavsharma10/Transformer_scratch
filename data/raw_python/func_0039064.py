def get_news_feed(self, symbol):
        """get_news_feed() uses the rss data table to get rss feeds under the Headlines and Financial Blogs headings on a typical stock page.
        """
        rss_url='http://finance.yahoo.com/rss/headline?s={0}'.format(symbol)
        response = self.select('rss',['title','link','description'],limit=2).where(['url','=',rss_url])
        return response