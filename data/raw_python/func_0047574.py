def as_html(self):
        """
        Returns the image of the Yahoo rss feed as an html string
        """

        return '<a href="{0}"><img height="{1}" width="{2}" src="{3}" alt="{4}"></a>'.format(
            self.link, self.height, self.width, self.url, self.title)