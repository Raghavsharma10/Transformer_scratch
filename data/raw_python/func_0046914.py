def request(self, *args, **kwargs):
        """
        Gets the request using the `_url` and converts it into a
        beautiful soup object.

        :param args:            The args to pass on to `requests`.
        :param kwargs:          The kwargs to pass on to `requests`.
        """
        response = requests.request(*args, **kwargs)
        return BeautifulSoup.BeautifulSoup(response.text, "html.parser")