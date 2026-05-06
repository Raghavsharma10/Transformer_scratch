def get_article_url(self):
        """
        Get the url of the TVP Info article itself, not the url of the preview with
        the 'Przejdź do artykułu' hyperlink.

        Returns:
            (str): Url of the article with the video.

        """
        html = requests.get(self.url).text
        soup = BeautifulSoup(html, 'lxml')
        div = soup.find('div', class_='more-back')

        if div:
            parsed_uri = urlparse(self.url)
            domain = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
            suffix = div.find('a', href=True)['href'].strip()
            article_url = domain + suffix
            return article_url
        else:
            return self.url