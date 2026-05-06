def _fetch(self, urls):
        """Perform bulk collection of data and return the content.

        Gathering responses is handled by the base class and uses futures to
        speed up the processing. Response data is saved inside a local variable
        to be used later in extraction.
        """
        responses = self._request_bulk(urls)
        for response in responses:
            try:
                soup = BeautifulSoup(response.content, 'html.parser',
                                     from_encoding="iso-8859-1")
                text = soup.get_text()
            except Exception:
                text = response.text
            self.data.append(text) # Opportunistic findings
        return responses