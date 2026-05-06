def _get_links(self, response):
        """
            Parses the response text and returns all the links in it.

        :param response: The Response object.
        """
        html_text = response.text.encode('utf-8')
        doc = document_fromstring(html_text)
        links = []
        for e in doc.cssselect('a'):
            links.append(e.get('href'))